# MediMind - Diabetic Retinopathy Detection

A deep learning project for automated diabetic retinopathy (DR) classification using EfficientNet-B0 with explainable AI through Grad-CAM visualization.

## 🏥 Project Overview

This project implements a computer vision system to classify diabetic retinopathy severity levels in retinal fundus images. The system uses transfer learning with EfficientNet-B0 and provides explainable predictions through Grad-CAM heatmaps.

### DR Classification Levels:

- **0**: No DR
- **1**: Mild DR
- **2**: Moderate DR
- **3**: Severe DR
- **4**: Proliferative DR

## 🔧 Architecture

- **Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Framework**: PyTorch
- **Image Processing**: OpenCV, Albumentations
- **Explainability**: Grad-CAM
- **Evaluation**: sklearn metrics, ROC curves, confusion matrices

## 📁 Project Structure

```
MediMind/
├── augmentation.py          # Data augmentation pipelines
├── model.py                 # DRModel class definition
├── preprocess.py           # Image preprocessing and dataset splitting
├── train.py                # Training script
├── evaluate_model.py       # Performance evaluation
├── gradcam_analysis.py     # Grad-CAM visualization
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.\.venv\Scripts\Activate.ps1

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your dataset in the `data/` folder with:

- Images: `data/*.png` or `data/*.jpg`
- Labels: `data/train.csv` (with columns: `id_code`, `diagnosis`)

```bash
# Preprocess images (resize to 224x224, apply CLAHE)
python preprocess.py --csv data/train.csv --image_dir data --clahe
```

### 3. Training

```bash
# Train the model
python train.py
```

Training will:

- Use 70% data for training, 15% for validation, 15% for testing
- Apply data augmentation (rotation, flip, brightness adjustment)
- Save the best model as `best_model.pth`

### 4. Evaluation

```bash
# Evaluate model performance
python evaluate_model.py
```

Generates:

- Confusion matrix
- ROC curves
- Per-class metrics
- Performance report

### 5. Explainability Analysis

```bash
# Generate Grad-CAM visualizations
python gradcam_analysis.py
```

Creates heatmaps showing which regions the model focuses on for predictions.

## 📊 Features

### Data Processing

- ✅ Image resizing (224×224)
- ✅ CLAHE contrast enhancement
- ✅ Data augmentation (rotation, flip, brightness)
- ✅ Stratified train/val/test split

### Model Training

- ✅ EfficientNet-B0 with transfer learning
- ✅ AdamW optimizer with learning rate scheduling
- ✅ Cross-entropy loss
- ✅ Early stopping based on validation accuracy

### Evaluation Metrics

- ✅ Accuracy, Precision, Recall, F1-score
- ✅ ROC-AUC curves (per-class and macro)
- ✅ Confusion matrix
- ✅ Class distribution analysis

### Explainable AI

- ✅ Grad-CAM heatmaps
- ✅ Visual explanations for predictions
- ✅ Analysis of correct vs incorrect predictions

## 🔬 Research Applications

This project is suitable for:

- Medical AI research papers
- Clinical decision support systems
- Educational purposes in medical imaging
- Diabetic retinopathy screening programs

### Key Research Contributions:

1. **Transfer Learning**: Demonstrates effective use of pre-trained models in medical imaging
2. **Explainable AI**: Provides interpretable predictions crucial for medical applications
3. **Comprehensive Evaluation**: Includes clinically relevant metrics and visualizations

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Albumentations
- scikit-learn
- Matplotlib
- Seaborn
- grad-cam
- timm
- tqdm

## 🎓 Usage for Research

### For Publications:

1. Use confusion matrices to discuss classification performance
2. Include ROC curves for clinical validation
3. Show Grad-CAM visualizations to demonstrate model focus on relevant lesions
4. Discuss failure cases using incorrect predictions

### Citation:

If you use this code in your research, please cite:

```
[Your research paper citation here]
```

## 📈 Expected Performance

Typical results on diabetic retinopathy datasets:

- **Accuracy**: 75-85%
- **AUC**: 0.80-0.90
- **F1-Score**: 0.70-0.80

_Performance varies based on dataset quality and size_

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- EfficientNet architecture from Google Research
- APTOS 2019 Blindness Detection dataset
- PyTorch and timm communities

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact 202204043.himanshupsk@student.xavier.ac.in.

---

**Note**: This is a research project. For clinical applications, ensure proper validation and regulatory compliance.
