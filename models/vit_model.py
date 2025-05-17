import torch
import torch.nn as nn
import os
import timm
import models_vit  # Assumed to be from MAE codebase and in PYTHONPATH
from util.pos_embed import interpolate_pos_embed  # Assumed from MAE utils

class ViTEncoder(nn.Module):
    """Vision Transformer (ViT) encoder for feature extraction"""
    
    def __init__(self, model_name='vit_base_patch16', finetune_path=None, 
                input_size=224, global_pool="avg", drop_path_rate=0.1, num_classes=1000):
        super().__init__()
        self.model_name = model_name
        
        # Validate ViT model type
        if 'vit_large' in model_name:
            print(f"Using ViT-Large architecture ({model_name}) for encoding")
        elif 'vit_base' in model_name:
            print(f"Using ViT-Base architecture ({model_name}) for encoding")
        else:
            print(f"Using {model_name} architecture for encoding")
            
        # Initialize the ViT model
        self.encoder = models_vit.__dict__[model_name](
            img_size=input_size,  # MAE ViT expects img_size, not list
            num_classes=num_classes,  # Can be dummy if head is replaced
            drop_path_rate=drop_path_rate,
            global_pool=global_pool
        )
        
        # Load weights - either custom or ImageNet pretrained
        if finetune_path and os.path.exists(finetune_path):
            print(f"Loading ViT weights from: {finetune_path}")
            checkpoint = torch.load(finetune_path, map_location='cpu', weights_only=False)
            vit_checkpoint_model = checkpoint.get('model', checkpoint)

            interpolate_pos_embed(self.encoder, vit_checkpoint_model)
            
            load_msg = self.encoder.load_state_dict(vit_checkpoint_model, strict=False)
            print(f"ViT weights loaded. Missing: {load_msg.missing_keys}, Unexpected: {load_msg.unexpected_keys}")
        else:
            print(f"Loading ImageNet pre-trained weights for {model_name}.")
            # Try to load ImageNet pretrained weights for ViT from timm
            try:
                # Convert MAE model name to timm format
                timm_model_name = model_name
                if 'vit_base_patch16' in model_name:
                    timm_model_name = 'vit_base_patch16_224'
                elif 'vit_large_patch16' in model_name:
                    timm_model_name = 'vit_large_patch16_224'
                elif 'vit_huge_patch14' in model_name:
                    timm_model_name = 'vit_huge_patch14_224'
                
                # Use timm to get pretrained ImageNet weights for the ViT model
                print(f"Attempting to load ImageNet weights for {timm_model_name} from timm")
                pretrained_vit = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0  # Get the model without classifier head
                )
                # Extract the state dict and load it to our model
                vit_state_dict = pretrained_vit.state_dict()
                
                # Interpolate position embeddings if needed
                interpolate_pos_embed(self.encoder, vit_state_dict)
                
                load_msg = self.encoder.load_state_dict(vit_state_dict, strict=False)
                print(f"ViT ImageNet weights loaded from timm. Missing: {load_msg.missing_keys}, Unexpected: {load_msg.unexpected_keys}")
            except (ImportError, RuntimeError, KeyError) as e:
                print(f"Could not load ImageNet weights for ViT: {e}")
                print("Initializing ViT with random weights.")
        
        # Determine feature dimension and remove head
        if hasattr(self.encoder, 'head'):
            self.feature_dim = self.encoder.head.in_features
            self.encoder.head = nn.Identity()
        elif hasattr(self.encoder, 'embed_dim'):
            self.feature_dim = self.encoder.embed_dim
        else:
            raise AttributeError("Cannot determine ViT feature dimension from encoder.")
    
    def forward(self, x):
        features = self.encoder(x)
        return features
    
    def get_param_groups(self, base_lr, weight_decay, layer_decay_rate=None):
        """Get parameter groups for optimizer with optional layer-wise lr decay"""
        param_groups = []
        
        if layer_decay_rate is not None and layer_decay_rate < 1.0 and hasattr(self.encoder, 'blocks'):
            # Layer-wise learning rate decay
            num_layers = len(self.encoder.blocks)
            layer_scales = list(layer_decay_rate ** (num_layers - i) for i in range(num_layers + 1))
            
            for name, param in self.encoder.named_parameters():
                if not param.requires_grad:
                    continue
                
                param_lr = base_lr
                param_wd = weight_decay
                
                if name.startswith('patch_embed'):
                    param_lr = base_lr * layer_scales[0]
                elif name.startswith('cls_token') or name.startswith('pos_embed'):
                    param_lr = base_lr * layer_scales[0]
                    param_wd = 0.0 
                elif name.startswith('blocks'):
                    try:
                        layer_id = int(name.split('.')[1])
                        param_lr = base_lr * layer_scales[layer_id + 1]
                    except:  # Should not happen with standard ViT block naming
                        pass 
                elif name.startswith('norm') or name.startswith('fc_norm'):  # Final norm layers
                    param_lr = base_lr * layer_scales[-1]
                
                if len(param.shape) == 1 or name.endswith(".bias"):
                    param_wd = 0.0
                
                param_groups.append({'params': [param], 'lr': param_lr, 'weight_decay': param_wd})
        else:
            # Standard parameter groups without layer-wise decay
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