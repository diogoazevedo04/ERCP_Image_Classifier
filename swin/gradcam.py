import os
import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import ERCPDataset
from transforms import get_val_transforms
from model import ERCPClassifier


def get_target_layer(model, model_name):
    """Devolve a(s) camada(s) alvo para Grad-CAM consoante a arquitetura."""
    name = model_name.lower()
    if 'convnext' in name:
        # último bloco do último stage
        return [model.backbone.stages[-1].blocks[-1]]
    if 'efficientnet' in name:
        if hasattr(model.backbone, 'conv_head'):
            return [model.backbone.conv_head]
        return [model.backbone.blocks[-1]]
    if 'swin' in name:
        return [model.backbone.layers[-1].blocks[-1].norm2]
    # fallback: última camada antes do pooling
    return [list(model.backbone.modules())[-2]]


def swin_reshape_transform(tensor, height=12, width=12):
    """Reshape de tokens (B, N, C) para (B, C, H, W) — necessário para transformers."""
    if tensor.dim() == 4:
        return tensor
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def denormalize(img_tensor):
    """De-normaliza um tensor ImageNet para [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean).clip(0, 1)
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_dir', default='processed_dataset')
    ap.add_argument('--split', default='test')
    ap.add_argument('--output_dir', default='gradcam_out')
    ap.add_argument('--n_per_class', type=int, default=8,
                    help='Quantas imagens guardar por classe')
    ap.add_argument('--method', default='gradcam++', choices=['gradcam', 'gradcam++'])
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt['model_name']
    classes = ckpt['classes']
    img_size = ckpt['img_size']

    model = ERCPClassifier(model_name=model_name, num_classes=len(classes),
                           pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    target_layers = get_target_layer(model, model_name)
    reshape_fn = swin_reshape_transform if 'swin' in model_name.lower() else None

    CamClass = GradCAMPlusPlus if args.method == 'gradcam++' else GradCAM
    cam = CamClass(model=model, target_layers=target_layers,
                   reshape_transform=reshape_fn)

    ds = ERCPDataset(os.path.join(args.data_dir, args.split),
                     transform=get_val_transforms(img_size))

    # Agrupa amostras por classe verdadeira
    per_class = {c: [] for c in range(len(classes))}
    for idx, (_, lbl) in enumerate(ds.samples):
        if len(per_class[lbl]) < args.n_per_class:
            per_class[lbl].append(idx)

    for cls_idx, indices in per_class.items():
        cls_name = classes[cls_idx]
        cls_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in indices:
            img_t, label = ds[i]
            input_tensor = img_t.unsqueeze(0).to(device)

            # Previsão
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            # Gera CAM para a classe prevista
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            rgb_img = denormalize(img_t)
            visualization = show_cam_on_image(rgb_img, grayscale_cam,
                                              use_rgb=True, image_weight=0.55)

            fname = os.path.basename(ds.samples[i][0]).replace('.png', '')
            out_name = f"{fname}_true-{cls_name}_pred-{classes[pred]}_p{probs[pred]:.2f}.png"
            out_path = os.path.join(cls_dir, out_name)

            # Side-by-side: original | heatmap
            original = (rgb_img * 255).astype(np.uint8)
            side = np.concatenate([original, visualization], axis=1)
            cv2.imwrite(out_path, cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

        print(f"[{cls_name}] {len(indices)} heatmaps guardados em {cls_dir}")

    print(f"\nGrad-CAM concluído. Resultados em: {args.output_dir}")


if __name__ == '__main__':
    main()