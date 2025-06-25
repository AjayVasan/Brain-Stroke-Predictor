# 🧠 Brain Stroke MRI Predictor/Classifier

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brain-stroke-predictor-ajayvasan.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

A custom-trained deep learning model to classify brain MRI images into various stroke types or detect the absence of stroke. Built with transfer learning using VGG19 and deployed using **Streamlit** for instant web-based predictions.

---

## 🌟 **Live Streamlit Demo**
[](Images/Images/Streamlit_Demo.png)

### 👉 **[🚀 Launch the Interactive Web App](https://brain-stroke-predictor-ajayvasan.streamlit.app/)**

**Experience the power of AI-driven medical imaging analysis:**
- 📤 **Drag & Drop** your MRI scan (PNG, JPEG)
- ⚡ **Instant Predictions** with confidence scores
- 📊 **Interactive Visualizations** and risk analysis
- 📱 **Mobile & Desktop** responsive interface
- 🎯 **Real-time Classification** of stroke types

> **No installation required** - just upload and predict!

---

## 📊 Project Overview

| Feature              | Details |
|----------------------|---------|
| **Model Type**       | Custom-trained VGG19 with Transfer Learning |
| **Input Format**     | Brain MRI scans (PNG, JPEG, JPG) |
| **Output Classes**   | Hemorrhagic Stroke, Ischemic Stroke, No Stroke |
| **Framework**        | TensorFlow + Keras |
| **Training Platform** | Google Colab (T4 GPU Runtime) |
| **Dataset**          | [Brain Stroke Dataset - Teknofest 2021](https://www.kaggle.com/datasets/shuvokumarbasakbd/brain-stroke-dataset-colorized-teknofest-2021) |
| **Deployment**       | [**Streamlit Cloud** ⭐](https://brain-stroke-predictor-ajayvasan.streamlit.app/) |
| **Model Repository** | [Hugging Face](https://huggingface.co/Ajay007001/Brain-Stroke-Prediction) |

---

## 🏗️ Model Architecture

- **Base Model**: Pretrained VGG19 (ImageNet weights)
- **Custom Head**: Dense layers for stroke classification
- **Data Augmentation**: Rotation, zoom, flip for robust training
- **Training Environment**: Google Colab T4 GPU
- **Fine-tuning**: Specialized for medical imaging

> 📓 **Training Notebook**: [`Brain_Stroke.ipynb`](./Brain_Stroke.ipynb)  
> 📸 **Training Insights**: [`Images/`](./Images) folder contains model performance visualizations

---

## ✨ **Streamlit App Features**

### 🎨 **User Experience**
- **Intuitive Interface**: Clean, professional medical-grade UI
- **File Upload**: Seamless drag-and-drop functionality  
- **Real-time Processing**: Instant analysis upon upload
- **Visual Feedback**: Progress indicators and loading states

### 📈 **Advanced Analytics**
- **Confidence Scores**: Percentage-based prediction reliability
- **Interactive Charts**: Bar graphs showing class probabilities
- **Risk Assessment**: Detailed prediction summaries
- **Medical Insights**: Educational information about stroke types

### 🔧 **Technical Capabilities**
- **Image Preprocessing**: Automatic resizing and normalization
- **Model Integration**: Direct connection to Hugging Face model
- **Error Handling**: Robust file validation and error messages
- **Performance Optimized**: Fast inference with caching

---

## 🚀 **Quick Start with Streamlit**

### Option 1: Use the Live App (Recommended)
Simply visit: **[brain-stroke-predictor-ajayvasan.streamlit.app](https://brain-stroke-predictor-ajayvasan.streamlit.app/)**

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/AjayVasan/Brain-Stroke-Predictor.git
cd Brain-Stroke-Predictor

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

> **Note**: Ensure you're using Python 3.10 as specified in `runtime.txt`

---

## 📁 Repository Structure

```
Brain-Stroke-Predictor/
│
├── 🚀 app.py                   # Main Streamlit application
├── 📓 Brain_Stroke.ipynb       # Model training notebook (Colab)
├── 📊 Images/                  # Training visualizations & insights
├── 📋 requirements.txt         # Python dependencies
├── 🐍 runtime.txt              # Python version specification
├── 📄 LICENSE                  # MIT License
├── 🐳 .devcontainer/           # Development container setup
└── 📖 README.md                # This file
```

---

## 🛠️ Technology Stack

**Frontend & Deployment:**
- **Streamlit** - Interactive web application framework
- **Streamlit Cloud** - Hosting and deployment platform

**Machine Learning:**
- **TensorFlow/Keras** - Deep learning framework
- **VGG19** - Convolutional neural network architecture
- **Hugging Face** - Model hosting and distribution

**Development:**
- **Google Colab** - GPU-accelerated training environment
- **Python 3.10** - Programming language
- **GitHub** - Version control and collaboration

---

## 🔗 Important Links

| Resource | Link |
|----------|------|
| 🌐 **Live Streamlit App** | [brain-stroke-predictor-ajayvasan.streamlit.app](https://brain-stroke-predictor-ajayvasan.streamlit.app/) |
| 🤗 **Hugging Face Model** | [Ajay007001/Brain-Stroke-Prediction](https://huggingface.co/Ajay007001/Brain-Stroke-Prediction) |
| 💻 **GitHub Repository** | [AjayVasan/Brain-Stroke-Predictor](https://github.com/AjayVasan/Brain-Stroke-Predictor) |
| 📊 **Dataset Source** | [Kaggle - Brain Stroke Dataset](https://www.kaggle.com/datasets/shuvokumarbasakbd/brain-stroke-dataset-colorized-teknofest-2021) |

---

## 🧑‍💻 Author

**Ajay Vasan**  
*Machine Learning Developer | Medical AI Enthusiast*

- 🌐 **Portfolio**: [ajayvasan.github.io/Portfolio](https://ajayvasan.github.io/Portfolio)
- 💼 **GitHub**: [@AjayVasan](https://github.com/AjayVasan)
- 🤗 **Hugging Face**: [@Ajay007001](https://huggingface.co/Ajay007001)
- 💼 **LinkedIn**: [@ajayvasan](https://www.linkedin.com/in/ajay-vasan)
---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Medical Disclaimer

**Important**: This application is designed for **educational and research purposes only**. It should **not be used as a substitute for professional medical diagnosis, advice, or treatment**. Always consult with qualified healthcare professionals for medical decisions.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/AjayVasan/Brain-Stroke-Predictor/issues).

---

## ⭐ Show Your Support

Give a ⭐ if this project helped you learn about medical AI and Streamlit deployment!

---

*Made with ❤️+🧠 and deployed with Streamlit*
