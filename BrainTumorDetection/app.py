import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
from model import BrainTumorCNN
from gradcam_util import GradCAM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load custom CNN model
model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load('brain_tumor_model.pth', map_location=device))
model.eval()

# GradCAM layer
target_layer = model.conv[6]  # Last conv layer in custom model
gradcam = GradCAM(model, target_layer)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Tumor Detector")
    st.markdown("This app uses a **Custom CNN model** trained on MRI scans to detect the presence of brain tumors.")
    st.markdown("### ℹ️ Model Info:")
    st.markdown("- Framework: **PyTorch**\n- Input size: **150x150**\n- Classes: **Tumor / No Tumor**")
    st.markdown("### 📁 Dataset Info:")
    st.markdown("- Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)")
    st.markdown("### 📌 How to Use:")
    st.markdown("1. Upload an MRI image\n2. View prediction\n3. Check Grad-CAM heatmap")
    st.markdown("---")
    st.markdown("💬 Built with ❤️ using Streamlit")

# Main Title
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an **MRI image** to detect the presence of a brain tumor using a deep learning model.")

# File upload
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Uploaded Image', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        class_names = ['No Tumor', 'Tumor']
        prediction = class_names[predicted.item()]

    st.markdown(f"### 🧪 Prediction: **:blue[{prediction}]**")

    # Grad-CAM
    cam = gradcam.generate(input_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    image_np = np.array(image.resize((150, 150))) / 255
    overlay = heatmap + image_np
    overlay = overlay / np.max(overlay)

    # Display GradCAM side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(image.resize((150, 150)), use_container_width=True)
    with col2:
        st.markdown("### 🔥 Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True)

    st.markdown("🧭 The heatmap highlights regions that contributed most to the model's prediction.")
else:
    st.warning("⚠️ Please upload an MRI image to get started.")
