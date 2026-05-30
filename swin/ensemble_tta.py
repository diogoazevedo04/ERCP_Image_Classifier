import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataset import ERCPDataset
from transforms import get_tta_transforms
from model import ERCPClassifier


@torch.no_grad()
def predict_with_tta(model, dataset_root, device, img_size, batch_size=16, num_workers=4):
    """Retorna probabilidades médias sobre todas as TTAs."""
    tta_list = get_tta_transforms(img_size)
    all_probs = None
    labels_ref = None
    for i, tfm in enumerate(tta_list):
        ds = ERCPDataset(dataset_root, transform=tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        probs_run, labels_run = [], []
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs_run.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_run.append(lbls.numpy())
        probs_run = np.concatenate(probs_run)
        labels_run = np.concatenate(labels_run)
        all_probs = probs_run if all_probs is None else all_probs + probs_run
        labels_ref = labels_run
        print(f"  TTA {i+1}/{len(tta_list)} concluída")
    return all_probs / len(tta_list), labels_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='processed_dataset')
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='Lista de paths para checkpoints .pt')
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='Pesos por modelo (opcional)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = args.weights or [1.0] * len(args.checkpoints)
    weights = np.array(weights) / sum(weights)

    ensemble_probs = None
    labels_ref = None
    class_names = None

    for path, w in zip(args.checkpoints, weights):
        print(f"\nA processar {path} (peso {w:.3f})")
        ckpt = torch.load(path, map_location=device)
        model = ERCPClassifier(model_name=ckpt['model_name'],
                               num_classes=len(ckpt['classes']),
                               pretrained=False).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        class_names = ckpt['classes']

        probs, labels = predict_with_tta(
            model,
            os.path.join(args.data_dir, args.split),
            device,
            img_size=ckpt['img_size'],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        ensemble_probs = probs * w if ensemble_probs is None else ensemble_probs + probs * w
        labels_ref = labels

    preds = ensemble_probs.argmax(axis=1)
    f1m = f1_score(labels_ref, preds, average='macro')
    print(f"\n=== ENSEMBLE+TTA {args.split.upper()} F1-macro: {f1m:.4f} ===")
    print(classification_report(labels_ref, preds, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels_ref, preds))

    np.save(f'ensemble_probs_{args.split}.npy', ensemble_probs)
    np.save(f'labels_{args.split}.npy', labels_ref)


if __name__ == '__main__':
    main()