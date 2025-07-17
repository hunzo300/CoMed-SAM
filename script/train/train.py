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


class CoMedSAM(nn.Module):
    def __init__(self, image_encoder_factory, mask_decoder, prompt_encoder, indicator):
        super().__init__()
        self.image_encoders = nn.ModuleList([image_encoder_factory() for _ in range(4)])
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.indicator = indicator

        self.conv1 = nn.Conv2d(256 * 4, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, images, box):
        B, n, C, H, W = images.shape
        image_embeddings = []

        for idx, encoder in enumerate(self.image_encoders):
            x = images[:, idx]
            if self.indicator[idx]:
                image_embeddings.append(encoder(x))
            else:
                with torch.no_grad():
                    image_embeddings.append(torch.zeros_like(encoder(x)))

        concat = torch.cat(image_embeddings, dim=1)
        x = self.gelu(self.conv1(concat))
        x = self.gelu(self.conv2(x))
        x = self.conv3(x)

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=images.device)[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=box_torch, masks=None)

        masks, _ = self.mask_decoder(
            image_embeddings=x,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)


parser = argparse.ArgumentParser()
parser.add_argument(
"-i",
"--tr_npy_path",
type=str,
default="/mnt/sda/minkyukim/CoMed-sam_dataset/IVDM/ivdm_npy_train_dataset_1024image",
help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
"--val_npy_path",
type=str,
default="/mnt/sda/minkyukim/CoMed-sam_dataset/IVDM/ivdm_npy_val_dataset_1024image",
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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2
    )
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
