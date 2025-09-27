import autorootcwd
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from segment_anything_CoMed import sam_model_registry
from script.IVDM3Seg.train.train import NpyDataset, CoMedSAM, join

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def inference_on_npy(data_root, npy_file=None, bbox_shift=0):
    dataset = NpyDataset(data_root, bbox_shift=bbox_shift)
    if npy_file:
        dataset.gt_path_files = [join(dataset.gt_path, npy_file)]
        print(f"Inference on: {npy_file}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sam_model = sam_model_registry["vit_b"](checkpoint="/SAM_PT/sam_vit_b_01ec64.pth")

    def create_image_encoder():
        return copy.deepcopy(sam_model.image_encoder).to(device)

    model = CoMedSAM(
        image_encoder_factory=create_image_encoder,
        mask_decoder=sam_model.mask_decoder.to(device),  
        prompt_encoder=sam_model.prompt_encoder.to(device), 
        indicator=[1,1,1,1]
    ).to(device)

    checkpoint = torch.load("/pth/IVDM3Seg/CoMed_SAM_IVDM3Seg.pth", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    os.makedirs("output_images", exist_ok=True)

    for step, (images, gt, bboxes, img_names) in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            pred_mask = model(images, bboxes.cpu().numpy())

        pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(0)
        pred_mask_np = np.clip(pred_mask_resized[0].cpu().numpy(), 0, 1)

        B, n, C, H, W = images.shape
        fig, axs = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

        for i in range(n):
            image_np = images[0, i].cpu().permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)
            axs[i].imshow(image_np)
            axs[i].set_title(f"Input Image {i+1}")
            axs[i].axis('off')

        axs[n].imshow(pred_mask_np, cmap='gray')
        axs[n].set_title("Predicted Mask")
        axs[n].axis('off')

        img_name_base = os.path.splitext(os.path.basename(img_names[0]))[0]
        save_path = f"output_images/{img_name_base}_pred.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    data_root = "/CoMed-sam_dataset/IVDM/ivdm_npy_test_dataset_1024image"
    npy_file = "sample_name"
    inference_on_npy(data_root, npy_file=npy_file)
