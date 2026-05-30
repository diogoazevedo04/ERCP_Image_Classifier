import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=384):
    pad = int(img_size * 1.15)
    return A.Compose([
        A.LongestMaxSize(max_size=pad),
        A.PadIfNeeded(min_height=pad, min_width=pad, border_mode=0, fill=0),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.70, 1.0), ratio=(0.85, 1.15)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, border_mode=0, p=0.7),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1),
                 shear=(-5, 5), border_mode=0, p=0.4),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.6),
        A.CoarseDropout(
            num_holes_range=(2, 6),
            hole_height_range=(int(img_size * 0.05), int(img_size * 0.12)),
            hole_width_range=(int(img_size * 0.05), int(img_size * 0.12)),
            fill=0, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=384):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=384):
    base = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
    ]
    norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [
        A.Compose(base + norm),
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),
        A.Compose(base + [A.VerticalFlip(p=1.0)] + norm),
        A.Compose(base + [A.Affine(rotate=(10, 10), border_mode=0, p=1.0)] + norm),
        A.Compose(base + [A.Affine(rotate=(-10, -10), border_mode=0, p=1.0)] + norm),
        A.Compose(base + [A.HorizontalFlip(p=1.0),
                          A.Affine(rotate=(10, 10), border_mode=0, p=1.0)] + norm),
    ]