import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def scan_split(split_dir):
    rows = []
    for cls in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(cls_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            rows.append({
                'path': path, 'class': cls,
                'height': h, 'width': w,
                'mean': float(img.mean()), 'std': float(img.std()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='processed_dataset')
    ap.add_argument('--output_dir', default='eda_out')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_df = []
    for split in ['train', 'val', 'test']:
        sp = os.path.join(args.data_dir, split)
        if not os.path.isdir(sp):
            continue
        df = scan_split(sp)
        df['split'] = split
        all_df.append(df)
    df = pd.concat(all_df, ignore_index=True)
    df.to_csv(os.path.join(args.output_dir, 'dataset_index.csv'), index=False)

    # Distribuição de classes por split
    pivot = df.groupby(['class', 'split']).size().unstack(fill_value=0)
    pivot['TOTAL'] = pivot.sum(axis=1)
    print("\n=== Distribuição de classes ===")
    print(pivot)
    pivot.to_csv(os.path.join(args.output_dir, 'class_distribution.csv'))

    # Gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.drop(columns=['TOTAL']).plot(kind='bar', ax=ax)
    ax.set_title('Distribuição de classes por split')
    ax.set_ylabel('# imagens')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'class_distribution.png'), dpi=120)
    plt.close()

    # Histograma de dimensões
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['width'], bins=40, color='steelblue')
    axes[0].set_title('Largura (px)')
    axes[1].hist(df['height'], bins=40, color='indianred')
    axes[1].set_title('Altura (px)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'image_sizes.png'), dpi=120)
    plt.close()

    # Estatísticas de intensidade por classe
    stats = df.groupby('class')[['mean', 'std', 'width', 'height']].agg(['mean', 'std'])
    stats.to_csv(os.path.join(args.output_dir, 'intensity_stats.csv'))
    print("\n=== Estatísticas por classe ===")
    print(stats)

    # Amostras visuais (4x4) por classe
    for cls in df['class'].unique():
        sub = df[df['class'] == cls].sample(min(16, len(df[df['class'] == cls])), random_state=0)
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax, (_, row) in zip(axes.flat, sub.iterrows()):
            img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        for ax in axes.flat[len(sub):]:
            ax.axis('off')
        fig.suptitle(f'Amostras — {cls}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'samples_{cls}.png'), dpi=110)
        plt.close()

    # Comparação visual: original vs CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    fig, axes = plt.subplots(len(df['class'].unique()), 2, figsize=(8, 3 * df['class'].nunique()))
    for i, cls in enumerate(sorted(df['class'].unique())):
        row = df[df['class'] == cls].iloc[0]
        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(img, cmap='gray'); axes[i, 0].set_title(f'{cls} — original'); axes[i, 0].axis('off')
        axes[i, 1].imshow(clahe.apply(img), cmap='gray'); axes[i, 1].set_title(f'{cls} — CLAHE'); axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'clahe_comparison.png'), dpi=120)
    plt.close()

    print(f"\nEDA concluída. Resultados em: {args.output_dir}")


if __name__ == '__main__':
    main()