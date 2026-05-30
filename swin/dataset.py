import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ERCPDataset(Dataset):
    """
    Dataset ERCP com CLAHE aplicado antes das augmentations.
    Espera estrutura: root/{class_name}/*.png
    """
    def __init__(self, root_dir, transform=None, apply_clahe=True, clahe_clip=2.5, clahe_grid=(8, 8)):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def _preprocess_clahe(self, img_bgr):
        # Converte para LAB, aplica CLAHE no canal L
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        merged = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # fallback via PIL caso o cv2 falhe
            img_pil = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if self.apply_clahe:
            img = self._preprocess_clahe(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        return img, label

    def get_labels(self):
        return [s[1] for s in self.samples]