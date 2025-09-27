import autorootcwd
import torch
import os
import copy
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from segment_anything_CoMed import sam_model_registry
from script.IVDM3Seg.train.train import CoMedSAM, NpyDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1
num_workers = 8
dataset_path = "/CoMed-sam_dataset/IVDM/ivdm_npy_test_dataset_1024image"
csv_file_path = "/output_csv/CoMed-SAM/phase_count_4.csv"

indicators = [
    [1, 1, 1, 1],
]

def calculate_metrics(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()
    pred_flat = pred_mask.view(-1).float()
    gt_flat = gt_mask.view(-1).float()
    intersection = torch.sum(pred_flat * gt_flat)
    union = torch.sum(pred_flat) + torch.sum(gt_flat) - intersection
    epsilon = 1e-7
    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (torch.sum(pred_flat) + torch.sum(gt_flat) + epsilon)
    return iou.item(), dice.item()

def test(checkpoint_path, indicator):
    dataset = NpyDataset(dataset_path)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    sam_model = sam_model_registry["vit_b"](checkpoint="SAM_PT/sam_vit_b_01ec64.pth")

    def create_image_encoder():
        return copy.deepcopy(sam_model.image_encoder).to(device)

    model = CoMedSAM(
        image_encoder_factory=create_image_encoder,
        mask_decoder=sam_model.mask_decoder.to(device),  
        prompt_encoder=sam_model.prompt_encoder.to(device),
        indicator=indicator, 
    ).to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    results = []

    for step, (images, gt, bboxes, img_names) in enumerate(tqdm(test_dataloader, desc=f"Testing checkpoint, mask {indicator}")):
        images = images.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            pred_mask = model(images, bboxes.cpu().numpy())
        # pred_mask_resized = F.interpolate(pred_mask, size=gt.shape[-2:], mode='nearest')
        iou, dice = calculate_metrics(pred_mask, gt)
        results.append({"iou": iou, "dice": dice})

    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    std_iou = np.std([r['iou'] for r in results])
    std_dice = np.std([r['dice'] for r in results])

    output_data = {
        "indicator": str(indicator),
        "average_iou": avg_iou,
        "std_iou": std_iou,
        "average_dice": avg_dice,
        "std_dice": std_dice
    }

    df = pd.DataFrame([output_data])
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

    print(f"Results for checkpoint, mask {indicator} saved to CSV.")

if __name__ == "__main__":
    checkpoint_path = f"/pth/IVDM3Seg/CoMed_SAM_IVDM3Seg.pth"
    print(f"Testing with checkpoint: {checkpoint_path}")
    for indicator in indicators:
        test(checkpoint_path, indicator)
