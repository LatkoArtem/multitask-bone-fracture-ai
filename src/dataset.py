import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset


class FracAtlasDataset(Dataset):
    """FracAtlas dataset: loads images from Fractured/Non_fractured dirs with matching masks."""

    BONE_CLASSES = ['hand', 'leg', 'hip', 'shoulder', 'mixed']

    def __init__(self, split_csv: str, dataset_csv: str, img_dir: str,
                 mask_dir: str, transform=None):
        split_df = pd.read_csv(split_csv)
        full_df  = pd.read_csv(dataset_csv)
        self.data_frame = (
            full_df[full_df['image_id'].isin(split_df['image_id'])]
            .reset_index(drop=True)
        )
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int):
        row      = self.data_frame.iloc[idx]
        img_name = str(row['image_id'])

        bone_tensor     = torch.tensor(row[self.BONE_CLASSES].values.astype(np.float32))
        fracture_tensor = torch.tensor(float(row['fractured']), dtype=torch.float32).unsqueeze(0)

        path_frac     = os.path.join(self.img_dir, 'Fractured',     img_name)
        path_non_frac = os.path.join(self.img_dir, 'Non_fractured', img_name)
        if os.path.exists(path_frac):
            img_path = path_frac
        elif os.path.exists(path_non_frac):
            img_path = path_non_frac
        else:
            raise FileNotFoundError(f"Image {img_name} not found in Fractured/Non_fractured")

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (raw > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        if self.transform is not None:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask'].unsqueeze(0)

        return image, bone_tensor, fracture_tensor, mask
