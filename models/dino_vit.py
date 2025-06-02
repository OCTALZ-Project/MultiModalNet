import torch
import torch.nn as nn
import os
from models.dinov2.models import vision_transformer as dino_vits

class DinoViTEncoder(nn.Module):
    """DINOv2 Vision Transformer (ViT) encoder for feature extraction"""
    def __init__(self, model_name='vit_base', finetune_path=None, input_size=224, drop_path_rate=0.1, num_classes=1000, **kwargs):
        super().__init__()
        self.model_name = model_name
        # DINOv2 ViT modelleri: vit_small, vit_base, vit_large, vit_giant2
        if model_name not in dino_vits.__dict__:
            raise ValueError(f"Unsupported DINOv2 ViT model: {model_name}")
        self.encoder = dino_vits.__dict__[model_name](
            img_size=input_size,
            drop_path_rate=drop_path_rate,
            **kwargs
        )
        # Ağırlık yükleme
        if finetune_path and os.path.exists(finetune_path):
            print(f"Loading DINOv2 ViT weights from: {finetune_path}")
            checkpoint = torch.load(finetune_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            load_msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"DINOv2 ViT weights loaded. Missing: {load_msg.missing_keys}, Unexpected: {load_msg.unexpected_keys}")
        # Özellik boyutu
        if hasattr(self.encoder, 'embed_dim'):
            self.feature_dim = self.encoder.embed_dim
        else:
            raise AttributeError("Cannot determine DINOv2 ViT feature dimension from encoder.")

    def forward(self, x):
        # DINOv2 returns a dictionary with different token types
        ret = self.encoder.forward_features(x)
        # Extract only the class token for feature extraction
        # ret["x_norm_clstoken"] contains the class token features [batch_size, embed_dim]
        features = ret["x_norm_clstoken"]
        return features

    def get_param_groups(self, base_lr, weight_decay, layer_decay_rate=None):
        # Standart parametre grupları (isteğe bağlı olarak layer-wise decay eklenebilir)
        params_no_decay = []
        params_decay = []
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                params_no_decay.append(param)
            else:
                params_decay.append(param)
        param_groups = [
            {'params': params_decay, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': params_no_decay, 'lr': base_lr, 'weight_decay': 0.0}
        ]
        return param_groups
