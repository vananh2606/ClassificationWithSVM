import torch
from torch.utils.data import Dataset
import os
import cv2


class MNISTFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []

        # Load ảnh và nhãn từ các thư mục con
        for label in range(10):
            label_folder = os.path.join(folder_path, str(label))
            if os.path.exists(label_folder):
                for img_name in os.listdir(label_folder):
                    img_path = os.path.join(label_folder, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
