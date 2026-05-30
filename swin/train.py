import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter

from dataset import ERCPDataset
from transforms import get_train_transforms, get_val_transforms, get_tta_transforms
from losses import FocalLoss
from model import ERCPClassifier


def build_sampler(labels):
    counts = Counter(labels)
    # sqrt-balancing: menos agressivo que 1/freq pura, evita oversampling extremo
    weights_per_class = {c: 1.0 / np.sqrt(cnt) for c, cnt in counts.items()}
    sample_weights = [weights_per_class[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
    x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        r = np.random.rand()
        with autocast('cuda'):
            if r < 0.3:
                mx, ya, yb, lam = mixup_data(imgs, labels, alpha=0.2)
                logits = model(mx)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            elif r < 0.6:
                mx, ya, yb, lam = cutmix_data(imgs, labels, alpha=1.0)
                logits = model(mx)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, tta_transforms=None, dataset_root=None,
             apply_clahe=True, batch_size=16, num_workers=2):
    """Se tta_transforms for fornecido, faz inferência com TTA."""
    model.eval()
    if tta_transforms is None:
        all_preds, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast('cuda'):
                logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return f1_score(all_labels, all_preds, average='macro'), all_preds, all_labels, None

    # TTA path
    all_probs = None
    labels_ref = None
    for tfm in tta_transforms:
        ds = ERCPDataset(dataset_root, transform=tfm, apply_clahe=apply_clahe)
        loader_t = DataLoader(ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
        probs_run, labels_run = [], []
        for imgs, lbls in loader_t:
            imgs = imgs.to(device, non_blocking=True)
            with autocast('cuda'):
                logits = model(imgs)
            probs_run.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_run.append(lbls.numpy())
        probs_run = np.concatenate(probs_run)
        labels_run = np.concatenate(labels_run)
        all_probs = probs_run if all_probs is None else all_probs + probs_run
        labels_ref = labels_run
    all_probs /= len(tta_transforms)
    preds = all_probs.argmax(axis=1)
    return f1_score(labels_ref, preds, average='macro'), preds, labels_ref, all_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='processed_dataset')
    ap.add_argument('--model_name', default='convnext_base.fb_in22k_ft_in1k')
    ap.add_argument('--img_size', type=int, default=384)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=5e-4)       # subido
    ap.add_argument('--backbone_lr_mult', type=float, default=0.2)  # subido (era 0.1)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--output_dir', default='checkpoints')
    ap.add_argument('--num_workers', type=int, default=2)   # reduzido (Colab só suporta 2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=18)
    ap.add_argument('--focal_gamma', type=float, default=1.5)  # reduzido (era 2.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = ERCPDataset(os.path.join(args.data_dir, 'train'),
                           transform=get_train_transforms(args.img_size))
    val_ds = ERCPDataset(os.path.join(args.data_dir, 'val'),
                         transform=get_val_transforms(args.img_size))
    test_ds = ERCPDataset(os.path.join(args.data_dir, 'test'),
                          transform=get_val_transforms(args.img_size))

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    sampler = build_sampler(train_ds.get_labels())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    counts = Counter(train_ds.get_labels())
    num_classes = len(train_ds.classes)
    # raiz quadrada do inverso (menos agressivo) — combina melhor com sqrt-sampler
    weights = torch.tensor([1.0 / np.sqrt(counts[i]) for i in range(num_classes)], dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    print(f"Class weights: {weights.tolist()}")

    model = ERCPClassifier(model_name=args.model_name, num_classes=num_classes,
                           pretrained=True, drop_rate=0.3, drop_path_rate=0.2).to(device)

    criterion = FocalLoss(alpha=weights.to(device), gamma=args.focal_gamma, label_smoothing=0.05)

    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params = [p for n, p in model.named_parameters() if 'head' in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * args.backbone_lr_mult},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')

    best_f1 = 0.0
    safe_name = args.model_name.replace('/', '_').replace('.', '_')
    best_path = os.path.join(args.output_dir, f'best_{safe_name}.pt')
    bad_epochs = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_f1, _, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | loss {train_loss:.4f} | val F1 {val_f1:.4f} | lr {lr_now:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state': model.state_dict(),
                        'classes': train_ds.classes,
                        'model_name': args.model_name,
                        'img_size': args.img_size}, best_path)
            print(f"  ✔ Saved (val F1 = {val_f1:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping no epoch {epoch+1}")
                break

    # Test com TTA já no fim
    print(f"\nA carregar o melhor checkpoint (val F1 = {best_f1:.4f})...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    tta_list = get_tta_transforms(args.img_size)
    test_f1, preds, labels, _ = evaluate(
        model, None, device,
        tta_transforms=tta_list,
        dataset_root=os.path.join(args.data_dir, 'test'),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"\n=== TEST F1-macro (com TTA): {test_f1:.4f} ===")
    print(classification_report(labels, preds, target_names=train_ds.classes, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == '__main__':
    main()