"""
ISBI_Loader 数据集加载类

INPUT:
- data_path: 数据集路径，包含模糊图像和对应的清晰图像.
- 模糊图像存放在 'blur' 文件夹下，清晰图像存放在 'sharp' 文件夹下.

OUTPUT:
- 返回一个元组 (image, label).
- image 是模糊图像.
- label 是清晰图像.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ISBI_Loader(Dataset):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'blur', '*.jpg'))

    def __getitem__(self, index: int) -> tuple:
        image_path = self.imgs_path[index]
        label_path = image_path.replace('blur', 'sharp')

        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = image / 255.0 if image.max() > 1 else image
        label = label / 255.0 if label.max() > 1 else label

        # (1, H, W)
        image = image[np.newaxis, :, :].astype(np.float32)
        label = label[np.newaxis, :, :].astype(np.float32)

        return image, label

    def __len__(self) -> int:
        return len(self.imgs_path)


if __name__ == "__main__":
    dataset = ISBI_Loader("dataset/train")
    print("数据个数:", len(dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    for images, labels in train_loader:
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        break