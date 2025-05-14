# Brain Tumor Detection using CNN and Grad-CAM 🧠

This project detects brain tumors in MRI images using a custom Convolutional Neural Network (CNN). It uses Grad-CAM to visualize model decisions and a Streamlit app to interactively test images. \
⚠️ You can train the model yourself using `train_model.py`


## Features
- CNN model built with PyTorch
- Grad-CAM visual explanations
- Streamlit web app interface
- Test with your own MRI images

## How to Run
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Dataset
- MRI scans from:\
 [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
 
- Organize your dataset as:\
dataset/\
├── train/\
│   ├── yes/\
│   └── no/\
├── val/\
└── test/

## Screenshots
![Screenshot (97)](https://github.com/user-attachments/assets/7864c969-7368-4eef-bdef-52a0a09b50d5)
![Screenshot (100)](https://github.com/user-attachments/assets/511490d3-514e-43d1-8bf6-e77abb78cd60)

## License
MIT
