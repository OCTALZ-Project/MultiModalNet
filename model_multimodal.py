import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
from models import ResNetEncoder, AlexNetEncoder, ViTEncoder

class ProjectionModelType(enum.Enum):
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'
    ALEXNET = 'alexnet'
    DINO_VIT = 'dino_vit'

class BscanModelType(enum.Enum):
    VIT_BASE = 'vit_base_patch16'
    VIT_LARGE = 'vit_large_patch16'
    VIT_HUGE = 'vit_huge_patch14'
    DINO_VIT_BASE = 'dino_vit_base'
    DINO_VIT_LARGE = 'dino_vit_large'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'
    ALEXNET = 'alexnet'

# Model loader helpers
def get_projection_encoder(model_type, projection_maps_model_finetune_path, projection_maps_model_dropout=0.0, input_size=224, **kwargs):
    if model_type == ProjectionModelType.ALEXNET:
        from models.alexnet_model import AlexNetEncoder
        return AlexNetEncoder(
            finetune_path=projection_maps_model_finetune_path,
            dropout=projection_maps_model_dropout
        )
    elif model_type in [ProjectionModelType.RESNET18, ProjectionModelType.RESNET34, ProjectionModelType.RESNET50, ProjectionModelType.RESNET101, ProjectionModelType.RESNET152]:
        from models.resnet_models import ResNetEncoder
        return ResNetEncoder(
            model_type=model_type.value,
            finetune_path=projection_maps_model_finetune_path,
            dropout=projection_maps_model_dropout
        )
    elif model_type == ProjectionModelType.DINO_VIT:
        from models.dino_vit import DinoViTEncoder
        return DinoViTEncoder(
            finetune_path=projection_maps_model_finetune_path,
            input_size=input_size
        )
    else:
        raise ValueError(f"Unsupported projection model: {model_type}")

def get_bscan_encoder(model_type, bscan_model_finetune_path, bscan_model_dropout=0.0, input_size=224, global_pool="avg", drop_path_rate=0.1, num_classes=1000, **kwargs):
    print(f"kwargs: {kwargs}")
    if model_type in [BscanModelType.RESNET50, BscanModelType.RESNET101, BscanModelType.RESNET152]:
        from models.resnet_models import ResNetEncoder
        return ResNetEncoder(
            model_type=model_type.value,
            finetune_path=bscan_model_finetune_path,
            dropout=bscan_model_dropout
        )
    elif model_type == BscanModelType.ALEXNET:
        from models.alexnet_model import AlexNetEncoder
        return AlexNetEncoder(
            finetune_path=bscan_model_finetune_path,
            dropout=bscan_model_dropout
        )
    elif model_type in [BscanModelType.DINO_VIT_BASE, BscanModelType.DINO_VIT_LARGE]:
        from models.dino_vit import DinoViTEncoder
        dino_model_name = model_type.value.replace('dino_vit_', 'vit_')
        return DinoViTEncoder(
            model_name=dino_model_name,
            finetune_path=bscan_model_finetune_path,
            input_size=input_size
        )
    else:
        from models.vit_model import ViTEncoder
        return ViTEncoder(
            model_name=model_type.value,
            finetune_path=bscan_model_finetune_path,
            input_size=input_size,
            global_pool=global_pool,
            drop_path_rate=drop_path_rate,
            num_classes=num_classes
        )

class MultiModalNet(nn.Module):
    def __init__(self, num_classes, projection_maps_model_finetune_path=None, bscan_model_finetune_path=None,
                 bscan_model_name='vit_base_patch16', bscan_model_global_pool="avg", bscan_model_drop_path_rate=0.1, bscan_model_input_size=224,
                 projection_maps_model='resnet50', projection_maps_model_dropout=0.0, bscan_model_dropout=0.0,
                 train_only_bscan=False, train_only_projection=False):
        super().__init__()
        self.num_classes = num_classes
        self.train_only_bscan = train_only_bscan
        self.train_only_projection = train_only_projection

        # Convert string to enum for safety
        proj_model_enum = ProjectionModelType(projection_maps_model)
        bscan_model_enum = BscanModelType(bscan_model_name)

        if train_only_bscan:
            self.bscan_encoder = get_bscan_encoder(
                bscan_model_enum,
                bscan_model_finetune_path,
                bscan_model_dropout=bscan_model_dropout,
                input_size=bscan_model_input_size,
                global_pool=bscan_model_global_pool,
                drop_path_rate=bscan_model_drop_path_rate,
                num_classes=self.num_classes
            )
            self.bscan_feature_dim = self.bscan_encoder.feature_dim
            self.classifier = nn.Linear(self.bscan_feature_dim, self.num_classes)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

        elif train_only_projection:
            self.projection_maps_encoder = get_projection_encoder(
                proj_model_enum,
                projection_maps_model_finetune_path,
                projection_maps_model_dropout=projection_maps_model_dropout,
                input_size=bscan_model_input_size
            )
            self.projection_maps_feature_dim = self.projection_maps_encoder.feature_dim
            self.classifier = nn.Linear(self.projection_maps_feature_dim, self.num_classes)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            if self.classifier.bias is not None:
                nn.init.zeros_(self.classifier.bias)

        else:
            self.projection_maps_encoder = get_projection_encoder(
                proj_model_enum,
                projection_maps_model_finetune_path,
                projection_maps_model_dropout=projection_maps_model_dropout,
                input_size=bscan_model_input_size
            )
            self.projection_maps_feature_dim = self.projection_maps_encoder.feature_dim
            self.bscan_encoder = get_bscan_encoder(
                bscan_model_enum,
                bscan_model_finetune_path,
                bscan_model_dropout=bscan_model_dropout,
                input_size=bscan_model_input_size,
                global_pool=bscan_model_global_pool,
                drop_path_rate=bscan_model_drop_path_rate,
                num_classes=self.num_classes
            )
            self.bscan_feature_dim = self.bscan_encoder.feature_dim
            self.fusion_dim = self.projection_maps_feature_dim + self.bscan_feature_dim
            self.fusion_fc = nn.Linear(self.fusion_dim, self.num_classes)
            nn.init.trunc_normal_(self.fusion_fc.weight, std=0.02)
            if self.fusion_fc.bias is not None:
                nn.init.zeros_(self.fusion_fc.bias)

        # Assign the correct forward function
        if self.train_only_bscan:
            self.forward = self._forward_bscan
        elif self.train_only_projection:
            self.forward = self._forward_projection
        else:
            self.forward = self._forward_both

    def _forward_bscan(self, octa_x, bscan_x):
        bscan_features = self.bscan_encoder(bscan_x)
        output = self.classifier(bscan_features)
        return output

    def _forward_projection(self, octa_x, bscan_x):
        octa_features = self.projection_maps_encoder(octa_x)
        output = self.classifier(octa_features)
        return output

    def _forward_both(self, octa_x, bscan_x):
        octa_features = self.projection_maps_encoder(octa_x)
        bscan_features = self.bscan_encoder(bscan_x)
        fused_features = torch.cat((octa_features, bscan_features), dim=1)
        output = self.fusion_fc(fused_features)
        return output

    def get_param_groups(self, base_lr, weight_decay, layer_decay_rate=None):
        param_groups = []
        from models import ViTEncoder
        if self.train_only_bscan:
            # Only B-scan encoder and classifier
            if isinstance(self.bscan_encoder, ViTEncoder):
                param_groups.extend(self.bscan_encoder.get_param_groups(base_lr, weight_decay, layer_decay_rate))
            else:
                param_groups.extend(self.bscan_encoder.get_param_groups(base_lr, weight_decay))
            # Classifier head
            classifier_params_no_decay = []
            classifier_params_decay = []
            for name, param in self.classifier.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    classifier_params_no_decay.append(param)
                else:
                    classifier_params_decay.append(param)
            param_groups.append({'params': classifier_params_decay, 'lr': base_lr * 2.0, 'weight_decay': weight_decay})
            param_groups.append({'params': classifier_params_no_decay, 'lr': base_lr * 2.0, 'weight_decay': 0.0})
        elif self.train_only_projection:
            # Only projection encoder and classifier
            param_groups.extend(self.projection_maps_encoder.get_param_groups(base_lr, weight_decay))
            classifier_params_no_decay = []
            classifier_params_decay = []
            for name, param in self.classifier.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    classifier_params_no_decay.append(param)
                else:
                    classifier_params_decay.append(param)
            param_groups.append({'params': classifier_params_decay, 'lr': base_lr * 2.0, 'weight_decay': weight_decay})
            param_groups.append({'params': classifier_params_no_decay, 'lr': base_lr * 2.0, 'weight_decay': 0.0})
        else:
            # Both encoders and fusion head
            param_groups.extend(self.projection_maps_encoder.get_param_groups(base_lr, weight_decay))
            if isinstance(self.bscan_encoder, ViTEncoder):
                param_groups.extend(self.bscan_encoder.get_param_groups(base_lr, weight_decay, layer_decay_rate))
            else:
                param_groups.extend(self.bscan_encoder.get_param_groups(base_lr, weight_decay))
            fusion_params_no_decay = []
            fusion_params_decay = []
            for name, param in self.fusion_fc.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias"):
                    fusion_params_no_decay.append(param)
                else:
                    fusion_params_decay.append(param)
            param_groups.append({'params': fusion_params_decay, 'lr': base_lr * 2.0, 'weight_decay': weight_decay})
            param_groups.append({'params': fusion_params_no_decay, 'lr': base_lr * 2.0, 'weight_decay': 0.0})
        return param_groups