import argparse
import cv2
import numpy as np
import torch
from transforms import get_val_transforms
from dataset import ERCPDataset  # só para reutilizar o CLAHE
from model import ERCPClassifier


def preprocess(path, img_size, apply_clahe=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if apply_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tfm = get_val_transforms(img_size)
    return tfm(image=img)['image']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--image', required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ERCPClassifier(model_name=ckpt['model_name'],
                           num_classes=len(ckpt['classes']),
                           pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    x = preprocess(args.image, ckpt['img_size']).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    print(f"\nImagem: {args.image}")
    for cls, p in zip(ckpt['classes'], probs):
        print(f"  {cls:15s} {p:.4f}")
    print(f"\n>>> Previsão: {ckpt['classes'][int(probs.argmax())]} (conf={probs.max():.3f})")


if __name__ == '__main__':
    main()