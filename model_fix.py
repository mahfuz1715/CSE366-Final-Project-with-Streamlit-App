import torch
import torch.nn as nn

def fix_state_dict_loading():
    """Fix the state dict loading issue by creating a compatibility layer"""
    
    # Load the saved state dict
    saved_state_dict = torch.load("custom_cnn_model.pth", map_location='cpu')
    
    # Create a mapping to fix the layer numbering
    new_state_dict = {}
    
    # Map the saved keys to expected keys
    key_mapping = {
        'features.9.weight': 'features.10.weight',
        'features.9.bias': 'features.10.bias',
        'features.9.running_mean': 'features.11.running_mean',
        'features.9.running_var': 'features.11.running_var',
        'features.12.weight': 'features.14.weight',
        'features.12.bias': 'features.14.bias',
        'features.13.weight': 'features.16.weight',
        'features.13.bias': 'features.16.bias',
        'features.13.running_mean': 'features.17.running_mean',
        'features.13.running_var': 'features.17.running_var',
    }
    
    # Create new state dict with correct keys
    for old_key, new_key in key_mapping.items():
        if old_key in saved_state_dict:
            new_state_dict[new_key] = saved_state_dict[old_key]
    
    # Add other keys that match directly
    for key in saved_state_dict:
        if key not in key_mapping:
            new_state_dict[key] = saved_state_dict[key]
    
    # Save the fixed state dict
    torch.save(new_state_dict, "custom_cnn_model_fixed.pth")
    print("Fixed state dict saved as custom_cnn_model_fixed.pth")
    
    return new_state_dict

def load_model_with_strict_false():
    """Alternative: Load model with strict=False to ignore mismatched keys"""
    from app import CustomCNN  # Import your model class
    
    model = CustomCNN()
    state_dict = torch.load("custom_cnn_model.pth", map_location='cpu')
    
    # Load with strict=False to ignore mismatched keys
    model.load_state_dict(state_dict, strict=False)
    
    print("Model loaded successfully with strict=False")
    return model

if __name__ == "__main__":
    fix_state_dict_loading()
