import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResNetEncoder, AlexNetEncoder, ViTEncoder

class MultiModalNet(nn.Module):
    def __init__(self, num_classes, projection_maps_model_finetune_path=None, bscan_model_finetune_path=None,
                 bscan_model_name='vit_base_patch16', bscan_model_global_pool="avg", bscan_model_drop_path_rate=0.1, bscan_model_input_size=224,
                 projection_maps_model='resnet50', projection_maps_model_dropout=0.0):
        super().__init__()
        self.num_classes = num_classes

        # Encoder for OCTA images - support ResNet50, ResNet101, or AlexNet
        if projection_maps_model == 'alexnet':
            self.projection_maps_encoder = AlexNetEncoder(
                finetune_path=projection_maps_model_finetune_path,
                dropout=projection_maps_model_dropout
            )
        elif projection_maps_model in ['resnet50', 'resnet101']:
            self.projection_maps_encoder = ResNetEncoder(
                model_type=projection_maps_model,
                finetune_path=projection_maps_model_finetune_path,
                dropout=projection_maps_model_dropout
            )
        else:
            raise ValueError(f"Unsupported model: {projection_maps_model}. Choose 'resnet50', 'resnet101', or 'alexnet'.")
        
        self.projection_maps_feature_dim = self.projection_maps_encoder.feature_dim

        # ViT encoder for B-scan images
        self.bscan_encoder = ViTEncoder(
            model_name=bscan_model_name,
            finetune_path=bscan_model_finetune_path,
            input_size=bscan_model_input_size,
            global_pool=bscan_model_global_pool,
            drop_path_rate=bscan_model_drop_path_rate,
            num_classes=self.num_classes
        )
        
        self.bscan_feature_dim = self.bscan_encoder.feature_dim

        # Fusion layer for both modalities
        self.fusion_dim = self.projection_maps_feature_dim + self.bscan_feature_dim
        self.fusion_fc = nn.Linear(self.fusion_dim, self.num_classes)

        nn.init.trunc_normal_(self.fusion_fc.weight, std=0.02)  # Common init for classifiers
        if self.fusion_fc.bias is not None:
            nn.init.zeros_(self.fusion_fc.bias)

    def forward(self, octa_x, bscan_x):
        # Extract features from both modalities
        octa_features = self.projection_maps_encoder(octa_x)
        bscan_features = self.bscan_encoder(bscan_x)
        
        # Fuse features and classify
        fused_features = torch.cat((octa_features, bscan_features), dim=1)
        output = self.fusion_fc(fused_features)
        return output

    def get_param_groups(self, base_lr, weight_decay, layer_decay_rate=None):
        param_groups = []
        
        # OCTA encoder parameters
        param_groups.extend(self.projection_maps_encoder.get_param_groups(base_lr, weight_decay))
        
        # B-scan encoder parameters with potential layer-wise decay
        param_groups.extend(self.bscan_encoder.get_param_groups(base_lr, weight_decay, layer_decay_rate))

        # Fusion FC parameters (classifier head)
        fusion_params_no_decay = []
        fusion_params_decay = []
        for name, param in self.fusion_fc.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                fusion_params_no_decay.append(param)
            else:
                fusion_params_decay.append(param)
        # Classifier head often benefits from a higher LR
        param_groups.append({'params': fusion_params_decay, 'lr': base_lr * 2.0, 'weight_decay': weight_decay}) 
        param_groups.append({'params': fusion_params_no_decay, 'lr': base_lr * 2.0, 'weight_decay': 0.0})
        
        return param_groups