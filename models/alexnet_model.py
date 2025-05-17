import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import AlexNet_Weights
import os

class AlexNetEncoder(nn.Module):
    """AlexNet based encoder with multi-layer LPIPS-style feature extraction"""
    
    def __init__(self, finetune_path=None, dropout=0.0):
        super().__init__()
        # Initialize alexnet without weights
        self.encoder = models.alexnet(weights=None)
        imagenet_weights = AlexNet_Weights.IMAGENET1K_V1
        
        # LPIPS-style feature extraction setup
        self.layer_indices = [1, 4, 7, 9, 11]  # Specific layers to extract features from
        self.lpips_weights = torch.tensor([0.051, 0.159, 0.215, 0.217, 0.358]).view(-1, 1, 1, 1)  # LPIPS weights
        
        # Calculate total dimension of all features after pooling
        self.feature_dim = 0
        temp_model = models.alexnet(weights=imagenet_weights)
        with torch.no_grad():
            temp_input = torch.randn(1, 3, 224, 224)
            features = []
            
            def hook_fn(module, input, output):
                features.append(output)
            
            hooks = []
            for idx in self.layer_indices:
                hooks.append(temp_model.features[idx].register_forward_hook(hook_fn))
            
            temp_model(temp_input)
            
            # Calculate pooled feature dimension
            for feat in features:
                # Apply adaptive pooling to get consistent size
                pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                self.feature_dim += pooled.view(-1).shape[0]
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Register hooks on the actual model
        self.features_list = []
        def get_hook_fn():
            def hook(module, input, output):
                self.features_list.append(output)
            return hook
        
        self.hooks = []
        for idx in self.layer_indices:
            self.hooks.append(self.encoder.features[idx].register_forward_hook(get_hook_fn()))
            
        print(f"Using AlexNet architecture with multi-layer features (dim={self.feature_dim})")
        
        # Load weights - either custom or ImageNet pretrained
        if finetune_path and os.path.exists(finetune_path):
            print(f"Loading custom AlexNet weights from: {finetune_path}")
            checkpoint = torch.load(finetune_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            
            # Remove classifier layers from state dict
            keys_to_remove = [k for k in state_dict if k.startswith('classifier.')]
            for k in keys_to_remove:
                del state_dict[k]
            
            load_msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"AlexNet custom weights loaded. Missing: {load_msg.missing_keys}, Unexpected: {load_msg.unexpected_keys}")
        else:
            print("Loading ImageNet pre-trained weights for AlexNet.")
            # Load weights from ImageNet pre-trained model
            imagenet_model = models.alexnet(weights=imagenet_weights)
            self.encoder.load_state_dict(imagenet_model.state_dict(), strict=False)
        
        # Replace classifier with identity since we don't need it
        self.encoder.classifier = nn.Identity()
        
        # Add dropout for features
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        if dropout > 0:
            print(f"Using dropout with rate {dropout} after feature extraction")
    
    def forward(self, x):
        self.features_list = []  # Reset features list
        self.encoder(x)  # Forward pass will collect features via hooks
        
        # Process features from each layer
        processed_features = []
        device = next(self.parameters()).device
        weights = self.lpips_weights.to(device)
        
        for i, feat in enumerate(self.features_list):
            # Normalize features channel-wise
            feat = F.normalize(feat, p=2, dim=1)
            # Weight the features
            feat = feat * weights[i]
            # Global average pooling and flatten
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(x.size(0), -1)
            processed_features.append(feat)
        
        # Concatenate all features
        features = torch.cat(processed_features, dim=1)
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
        
    def __del__(self):
        # Clean up hooks when model is deleted
        for hook in self.hooks:
            hook.remove() 