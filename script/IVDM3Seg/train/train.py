import os, random, argparse, copy, shutil, glob
from datetime import datetime
import autorootcwd

join = os.path.join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.transform import resize
import matplotlib.pyplot as plt
import monai
from tqdm import tqdm
from segment_anything_CoMed import sam_model_registry

# 환경 설정
torch.manual_seed(2023)
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=0, dropout=False):
        self.dropout = dropout
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")

        self.dataset_type = "ivdm"  
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            f for f in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(f).split('_')[0] + ".npy"))
        ]

        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index]).split('_')[0] + '.npy'
        img_4ch = np.load(os.path.join(self.img_path, img_name), "r")  # (4, H, W)
        img_4ch_resized = np.stack([resize(img_4ch[i], (1024, 1024), anti_aliasing=True) for i in range(4)])
        img_rgb_4 = np.stack([np.stack([img_4ch_resized[i]]*3, axis=0) for i in range(4)])  # (4, 3, 1024, 1024)

        if self.dropout:
            zero_indices = random.sample(range(4), random.randint(0, 3))
            for i in zero_indices:
                img_rgb_4[i] = 0.0

        gt = np.load(self.gt_path_files[index], "r")
        gt_1024 = resize(gt, (1024, 1024), anti_aliasing=False, preserve_range=True).astype(np.uint8)
        gt_1024 = (gt_1024 > 0).astype(np.uint8)

        label_ids = np.unique(gt_1024)[1:]
        if len(label_ids) == 0:
            return self.__getitem__((index + 1) % len(self.gt_path_files))

        gt2D = np.uint8(gt_1024 == random.choice(label_ids.tolist()))
        y, x = np.where(gt2D > 0)
        x_min, x_max = max(0, np.min(x) - random.randint(0, self.bbox_shift)), min(1024, np.max(x) + random.randint(0, self.bbox_shift))
        y_min, y_max = max(0, np.min(y) - random.randint(0, self.bbox_shift)), min(1024, np.max(y) + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img_rgb_4).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )


def get_2d_sincos_pos_embed(h, w, dim, device):

    assert dim % 4 == 0, "pos embed dim must be divisible by 4"
    dim_quarter = dim // 4

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    yy = yy.flatten().float()   # [hw]
    xx = xx.flatten().float()   # [hw]

    omega = torch.arange(dim_quarter, device=device).float() / dim_quarter
    omega = 1.0 / (10000 ** omega)  # [dim/4]

    out_y = torch.einsum("n,d->nd", yy, omega)  # [hw, dim/4]
    out_x = torch.einsum("n,d->nd", xx, omega)  # [hw, dim/4]

    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=1)  # [hw, dim/2]
    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=1)  # [hw, dim/2]
    pos = torch.cat([pos_y, pos_x], dim=1)  # [hw, dim]
    return pos.unsqueeze(0)  # [1, hw, dim]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, mlp_ratio=4.0, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout, batch_first=True)
        self.drop1 = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(proj_dropout)

    def forward(self, x):
        # x: [B, HW, C]
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x

class CoMedSAM(nn.Module):

    def __init__(self, image_encoder_factory, mask_decoder, prompt_encoder, indicator,
                 d_model=256, nhead=8, mlp_ratio=4.0, proj_dropout=0.0):
        super().__init__()
        self.image_encoder_factory = image_encoder_factory
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.indicator = indicator
        self.d_model = d_model

        self.image_encoders = nn.ModuleList([self.image_encoder_factory() for _ in range(4)])

        self.conv1 = nn.Conv2d(256 * 4, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.act = nn.GELU()

        self.tr_proj_in = nn.Conv2d(256 * 4, d_model, kernel_size=1)
        self.tr_block1 = TransformerBlock(d_model=d_model, nhead=nhead, mlp_ratio=mlp_ratio, proj_dropout=proj_dropout)
        self.tr_block2 = TransformerBlock(d_model=d_model, nhead=nhead, mlp_ratio=mlp_ratio, proj_dropout=proj_dropout)
        self.tr_block3 = TransformerBlock(d_model=d_model, nhead=nhead, mlp_ratio=mlp_ratio, proj_dropout=proj_dropout)

        self.out_norm = nn.GroupNorm(8, d_model)

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.parameters():
            p.requires_grad = False

    def forward(self, images, box):
        B, n, C, H, W = images.shape

        splits = torch.split(images, 1, dim=1)
        embs = []
        for idx, enc in enumerate(self.image_encoders):
            x = splits[idx].squeeze(1)  # [B,C,H,W]
            if self.indicator[idx] == 1:
                e = enc(x)
            else:
                with torch.no_grad():
                    e = enc(x)
                e = torch.zeros_like(e)
            embs.append(e)

        concat = torch.cat(embs, dim=1)  # [B,1024,H',W']
        Bh, Bw = concat.shape[-2], concat.shape[-1]


        conv_out = self.act(self.conv1(concat))
        conv_out = self.act(self.conv2(conv_out))
        conv_out = self.act(self.conv3(conv_out))  # [B,256,H',W']


        tr = self.tr_proj_in(concat)              # [B,256,H',W']
        tr_flat = tr.flatten(2).transpose(1, 2)   # [B,HW,256]
        pos = get_2d_sincos_pos_embed(Bh, Bw, self.d_model, device=tr.device)
        tr_flat = tr_flat + pos

        tr_flat = self.tr_block1(tr_flat)
        tr_flat = self.tr_block2(tr_flat)
        tr_flat = self.tr_block3(tr_flat)
        tr_out = tr_flat.transpose(1, 2).reshape(B, self.d_model, Bh, Bw)

        fused = conv_out + tr_out
        fused = self.out_norm(fused)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=images.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B,1,4)
            sparse_emb, dense_emb = self.prompt_encoder(points=None, boxes=box_torch, masks=None)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=fused,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

        ori_res_masks = F.interpolate(low_res_masks, size=(H, W), mode="bilinear", align_corners=False)
        return ori_res_masks


parser = argparse.ArgumentParser()
parser.add_argument(
"-i",
"--tr_npy_path",
type=str,
default="/CoMed-sam_dataset/IVDM/ivdm_npy_train_dataset_1024image",
help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
"--val_npy_path",
type=str,
default="/CoMed-sam_dataset/IVDM/ivdm_npy_val_dataset_1024image",
help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
"-checkpoint", type=str, default="SAM_PT/sam_vit_b_01ec64.pth"
)
parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


def main():

    device = torch.device(args.device)
    save_path = f"./pth"
    os.makedirs(save_path, exist_ok=True)

    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)

    model = CoMedSAM(
        image_encoder_factory=lambda: copy.deepcopy(sam_model.image_encoder).to(device),
        mask_decoder=sam_model.mask_decoder.to(device),
        prompt_encoder=sam_model.prompt_encoder.to(device),
        indicator=[1, 1, 1, 1],
    ).to(device)

    img_enc_params = []
    for encoder in model.image_encoders:
        for name, param in encoder.named_parameters():
            if "adapter" in name.lower():  
                param.requires_grad = True
                img_enc_params.append(param)
            else:
                param.requires_grad = False

    conv_params = (
        list(model.conv1.parameters())
        + list(model.conv2.parameters())
        + list(model.conv3.parameters())
    )

    optimizer = torch.optim.AdamW(
        img_enc_params + conv_params, lr=1e-4, weight_decay=1e-2
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {total_params}, Trainable: {trainable_params}")


    loss_fn_dice = monai.losses.DiceLoss(sigmoid=True)
    loss_fn_ce = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(NpyDataset(args.tr_npy_path, dropout=True), batch_size=1, shuffle=True)
    val_loader = DataLoader(NpyDataset(args.val_npy_path, dropout=True), batch_size=1)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y, b, _ in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x, b.numpy())
            loss = loss_fn_dice(pred, y) + loss_fn_ce(pred, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, b, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x, b.numpy())
                loss = loss_fn_dice(pred, y) + loss_fn_ce(pred, y.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(save_path, f"Comed_{epoch}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))


if __name__ == "__main__":
    main()
