import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4CAF50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .mango-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #FF9800;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
    }
    .model-selector {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #2196F3;
        margin-bottom: 2rem;
    }
    .progress-bar {
        height: 8px;
        background: #E0E0E0;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF6B35, #FF9800);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .mango-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        color: #FF6B35;
    }
    .stats-card {
        background: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Class names & device
# =========================
CLASS_NAMES = [
    "🍋 Amrapali Mango",
    "🍌 Banana Mango", 
    "🌟 Chaunsa Mango",
    "👑 Fazli Mango",
    "🌿 Haribhanga Mango",
    "✨ Himsagar Mango"
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Custom CNN definition
# =========================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =========================
# Model loaders
# =========================
def load_custom_cnn():
    model = CustomCNN(NUM_CLASSES).to(DEVICE)
    
    try:
        state_dict = torch.load("custom_cnn_model.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.warning("⚠️ Model architecture mismatch detected. Loading with partial weights...")
        state_dict = torch.load("custom_cnn_model.pth", map_location=DEVICE)
        model_dict = model.state_dict()
        filtered_dict = {}
        
        for key in model_dict.keys():
            if key in state_dict:
                filtered_dict[key] = state_dict[key]
            else:
                filtered_dict[key] = model_dict[key]
        
        model.load_state_dict(filtered_dict, strict=False)
    
    model.eval()
    return model

def load_resnet50():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_resnet50.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_densenet121():
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_densenet121.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_efficientnet_b0():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_efficientnet_b0.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_mobilenetv2():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("transfer_learning_mobilenetv2.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# =========================
# Target layer selector
# =========================
def get_target_layer(model):
    if hasattr(model, "layer4"):  # ResNet
        return model.layer4[-1]
    elif hasattr(model, "features"):  # VGG, DenseNet, EfficientNet, Custom CNN
        conv_layers = [m for m in model.features if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    return module
            raise ValueError("No convolutional layers found in the model.")
        return conv_layers[-1]
    else:
        raise ValueError("Target layer not found")

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# =========================
# Streamlit UI - Redesigned
# =========================
def main():
    # Header Section
    st.markdown('<div class="main-header">🍃 Bangladeshi Mango Leaf Classifier</div>', unsafe_allow_html=True)
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        st.markdown('### 🎯 Choose Model')
        model_choice = st.selectbox(
            "Select a model architecture:",
            ("Custom CNN", "ResNet-50", "DenseNet121", "EfficientNet-B0", "MobileNetV2"),
            help="Different models may provide varying accuracy and speed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image Upload
        st.markdown('### 📸 Upload Image')
        uploaded_file = st.file_uploader(
            "Choose a mango leaf image", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a Bangladeshi mango leaf for classification"
        )
        
        # About Section
        with st.expander("ℹ️ About this App"):
            st.markdown("""
            **Bangladeshi Mango Leaf Classifier** 🍃
            
            This app uses deep learning to classify different varieties of Bangladeshi mango leaves:
            
            - 🍋 Amrapali Mango
            - 🍌 Banana Mango  
            - 🌟 Chaunsa Mango
            - 👑 Fazli Mango
            - 🌿 Haribhanga Mango
            - ✨ Himsagar Mango
            
            **Features:**
            - Multiple model architectures
            - Explainable AI visualizations
            - Real-time predictions
            - Professional interface
            """)
    
    # Main Content Area
    if uploaded_file:
        # Display uploaded image immediately
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="mango-card">', unsafe_allow_html=True)
            st.markdown('### 📷 Uploaded Image')
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Your uploaded mango leaf image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown('### 📊 Image Info')
            st.write(f"**Format:** {img.format}")
            st.write(f"**Size:** {img.size[0]}×{img.size[1]} pixels")
            st.write(f"**Mode:** {img.mode}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Load selected model with progress
        with st.spinner(f"🚀 Loading {model_choice} model..."):
            if model_choice == "Custom CNN":
                model = load_custom_cnn()
            elif model_choice == "ResNet-50":
                model = load_resnet50()
            elif model_choice == "DenseNet121":
                model = load_densenet121()
            elif model_choice == "EfficientNet-B0":
                model = load_efficientnet_b0()
            elif model_choice == "MobileNetV2":
                model = load_mobilenetv2()
        
        # Prediction Section
        st.markdown('<div class="mango-card">', unsafe_allow_html=True)
        st.markdown('### 🔮 Prediction Results')
        
        with st.spinner("🧠 Analyzing the image..."):
            tensor_img = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(tensor_img)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                
                # Get top 3 predictions
                top3_idxs = np.argsort(probs)[-3:][::-1]
                top3_classes = [CLASS_NAMES[i] for i in top3_idxs]
                top3_probs = [probs[i] * 100 for i in top3_idxs]
                
                pred_idx = top3_idxs[0]
                pred_class = top3_classes[0]
                pred_prob = top3_probs[0]
        
        # Display predictions with progress bars
        st.markdown(f"#### 🎯 Top Prediction: **{pred_class.split(' ')[1]}**")
        
        for i in range(3):
            with st.container():
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{top3_classes[i]}")
                with col_b:
                    st.write(f"{top3_probs[i]:.1f}%")
                
                # Progress bar
                st.markdown('<div class="progress-bar">', unsafe_allow_html=True)
                st.markdown(f'<div class="progress-fill" style="width: {top3_probs[i]}%"></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # XAI Visualizations
        st.markdown('<div class="mango-card">', unsafe_allow_html=True)
        st.markdown('### 🔍 Explainable AI Visualizations')
        
        # Prepare CAM methods
        rgb_img = np.array(img.resize((224, 224))) / 255.0
        target_layers = [get_target_layer(model)]
        
        # Generate CAM visualizations
        with st.spinner("🖼️ Generating visual explanations..."):
            gradcam = GradCAM(model=model, target_layers=target_layers)
            gradcam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
            eigencam = EigenCAM(model=model, target_layers=target_layers)
            ablationcam = AblationCAM(model=model, target_layers=target_layers)
            
            grayscale_gradcam = gradcam(input_tensor=tensor_img)[0]
            grayscale_gradcam_pp = gradcam_pp(input_tensor=tensor_img)[0]
            grayscale_eigencam = eigencam(input_tensor=tensor_img)[0]
            grayscale_ablationcam = ablationcam(input_tensor=tensor_img)[0]
            
            vis_gradcam = show_cam_on_image(rgb_img, grayscale_gradcam, use_rgb=True)
            vis_gradcam_pp = show_cam_on_image(rgb_img, grayscale_gradcam_pp, use_rgb=True)
            vis_eigencam = show_cam_on_image(rgb_img, grayscale_eigencam, use_rgb=True)
            vis_ablationcam = show_cam_on_image(rgb_img, grayscale_ablationcam, use_rgb=True)
        
        # Display CAM visualizations
        st.markdown("#### 📊 CAM Methods Comparison")
        cols = st.columns(4)
        cam_images = [vis_gradcam, vis_gradcam_pp, vis_eigencam, vis_ablationcam]
        cam_titles = ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Ablation-CAM"]
        
        for col, title, img_cam in zip(cols, cam_titles, cam_images):
            with col:
                st.image(img_cam, caption=title, use_container_width=True)
        
        # LIME visualization
        st.markdown("#### 🎨 LIME Explanation")
        
        def batch_predict(images):
            batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
            logits = model(batch.to(DEVICE))
            return torch.softmax(logits, dim=1).detach().cpu().numpy()
        
        with st.spinner("🎨 Generating LIME explanation..."):
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                np.array(img), batch_predict, top_labels=1, hide_color=0, num_samples=100
            )
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
            )
            lime_vis = mark_boundaries(temp / 255.0, mask)
        
        # Display LIME
        col3, col4 = st.columns(2)
        with col3:
            st.image(np.array(img.resize((224, 224))), caption="Original Image", use_container_width=True)
        with col4:
            st.image(lime_vis, caption="LIME Explanation", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Success message
        st.success("✅ Analysis complete! All visualizations generated successfully.")
    
    else:
        # Welcome message when no image uploaded
        st.markdown('<div class="mango-card">', unsafe_allow_html=True)
        st.markdown("""
        ## 👋 Welcome to the Bangladeshi Mango Leaf Classifier!
        
        **Get started in 3 simple steps:**
        
        1. **📸 Upload** - Choose a mango leaf image using the sidebar
        2. **🎯 Select** - Pick your preferred model architecture  
        3. **🔍 Analyze** - Get instant predictions and AI explanations
        
        **Supported mango varieties:**
        - 🍋 Amrapali - Sweet and aromatic
        - 🍌 Banana - Unique banana-like shape
        - 🌟 Chaunsa - Premium quality
        - 👑 Fazli - King of mangoes
        - 🌿 Haribhanga - Traditional variety
        - ✨ Himsagar - Famous for taste
        
        *Upload an image to begin your analysis!*
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sample images placeholder
        st.markdown("### 📸 Sample Images Gallery")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Amrapali Mango Leaf")
        with col2:
            st.info("Fazli Mango Leaf")
        with col3:
            st.info("Himsagar Mango Leaf")

if __name__ == "__main__":
    main()
