import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
import os

class ResNetEncoder(nn.Module):
    """ResNet based encoder for feature extraction"""
    
    def __init__(self, model_type='resnet50', finetune_path=None, dropout=0.0):
        super().__init__()
        self.model_type = model_type
        
        # Initialize the model architecture
        if model_type == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            imagenet_weights = ResNet50_Weights.IMAGENET1K_V1
            self.feature_dim = self.encoder.fc.in_features
            print("Using ResNet50 architecture for encoding")
        elif model_type == 'resnet101':
            self.encoder = models.resnet101(weights=None)
            imagenet_weights = ResNet101_Weights.IMAGENET1K_V1
            self.feature_dim = self.encoder.fc.in_features
            print("Using ResNet101 architecture for encoding")
        elif model_type == 'resnet152':
            self.encoder = models.resnet152(weights=None)
            imagenet_weights = ResNet152_Weights.IMAGENET1K_V1
            self.feature_dim = self.encoder.fc.in_features
            print("Using ResNet152 architecture for encoding")
        else:
            raise ValueError(f"Unsupported ResNet model: {model_type}. Choose 'resnet50', 'resnet101', or 'resnet152'.")
            
        # Load weights - either custom or ImageNet pretrained
        if finetune_path and os.path.exists(finetune_path):
            print(f"Loading custom {model_type} weights from: {finetune_path}")
            checkpoint = torch.load(finetune_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # Remove classifier layers from state dict
            keys_to_remove = [k for k in state_dict if k.startswith('fc.')]
            for k in keys_to_remove:
                del state_dict[k]
            
            load_msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"{model_type} custom weights loaded. Missing: {load_msg.missing_keys}, Unexpected: {load_msg.unexpected_keys}")
        else:
            print(f"Loading ImageNet pre-trained weights for {model_type}.")
            # Load weights from ImageNet pre-trained model
            imagenet_model = models.__dict__[model_type](weights=imagenet_weights)
            self.encoder.load_state_dict(imagenet_model.state_dict(), strict=False)
        
        # Replace classifier with identity
        self.encoder.fc = nn.Identity()
        
        # Add dropout for features
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        if dropout > 0:
            print(f"Using dropout with rate {dropout} after feature extraction")
    
    def forward(self, x):
        features = self.encoder(x)
        features = self.dropout(features)
        return features
    
    def get_param_groups(self, base_lr, weight_decay):
        """Get parameter groups for optimizer"""
        params_no_decay = []
        params_decay = []
        
        for name, param in self.encoder.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):  # Biases and 1D params (like BN)
                params_no_decay.append(param)
            else:
                params_decay.append(param)
                
        param_groups = [
            {'params': params_decay, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': params_no_decay, 'lr': base_lr, 'weight_decay': 0.0}
        ]
        
        return param_groups