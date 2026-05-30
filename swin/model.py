import torch
import torch.nn as nn
import timm

class ERCPClassifier(nn.Module):
    """
    Wrapper genérico para modelos do timm com dropout extra e cabeça customizada.
    """
    def __init__(self, model_name='convnext_base.fb_in22k_ft_in1k', num_classes=4,
                 pretrained=True, drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,           # remove a cabeça original
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            global_pool='avg',
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def forward_features(self, x):
        return self.backbone.forward_features(x)