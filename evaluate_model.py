import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

# Import our custom modules
from model import DRModel
from augmentation import val_transform

# ================================
# Configuration
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
TEST_DIR = "data_processed/test"
OUTPUT_DIR = "results/metrics"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = 5
BATCH_SIZE = 32

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Dataset Class for Testing
# ================================
class TestDataset(Dataset):
    """Dataset class for testing."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = [int(os.path.basename(os.path.dirname(path))) for path in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

# ================================
# Model Loading
# ================================
def load_trained_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return None
    
    model = DRModel(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    return model

# ================================
# Inference Functions
# ================================
def run_inference(model, test_loader):
    """Run inference on the test set and collect predictions."""
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    print("🔄 Running inference on test set...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

# ================================
# Metrics Calculation
# ================================
def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC-AUC (multiclass)
    try:
        y_true_binarized = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        if y_true_binarized.shape[1] == NUM_CLASSES:
            metrics['roc_auc_macro'] = roc_auc_score(y_true_binarized, y_prob, average='macro', multi_class='ovr')
            metrics['roc_auc_per_class'] = roc_auc_score(y_true_binarized, y_prob, average=None, multi_class='ovr')
        else:
            metrics['roc_auc_macro'] = None
            metrics['roc_auc_per_class'] = None
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC: {e}")
        metrics['roc_auc_macro'] = None
        metrics['roc_auc_per_class'] = None
    
    return metrics

# ================================
# Visualization Functions
# ================================
def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Diabetic Retinopathy Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Confusion matrix saved to {save_path}")

def plot_roc_curves(y_true, y_prob, save_path):
    """Plot ROC curves for each class."""
    if y_prob is None:
        print("❌ Cannot plot ROC curves: probabilities not available")
        return
    
    # Binarize the output
    y_true_binarized = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    
    if y_true_binarized.shape[1] != NUM_CLASSES:
        print("❌ Cannot plot ROC curves: not all classes present in test set")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot ROC curve for each class
    for i in range(NUM_CLASSES):
        if np.sum(y_true_binarized[:, i]) > 0:  # Only plot if class exists in test set
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            auc_score = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, linewidth=2, label=f'{CLASS_NAMES[i]} (AUC = {auc_score:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Diabetic Retinopathy Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ ROC curves saved to {save_path}")

def plot_class_distribution(y_true, y_pred, save_path):
    """Plot distribution of true vs predicted classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # True class distribution
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    ax1.bar([CLASS_NAMES[i] for i in unique_true], counts_true, color='skyblue', alpha=0.7)
    ax1.set_title('True Class Distribution')
    ax1.set_ylabel('Number of Images')
    ax1.tick_params(axis='x', rotation=45)
    
    # Predicted class distribution
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    ax2.bar([CLASS_NAMES[i] for i in unique_pred], counts_pred, color='lightcoral', alpha=0.7)
    ax2.set_title('Predicted Class Distribution')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Class distribution plot saved to {save_path}")

def plot_per_class_metrics(metrics, save_path):
    """Plot per-class precision, recall, and F1-score."""
    classes = CLASS_NAMES
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Per-class metrics plot saved to {save_path}")

# ================================
# Report Generation
# ================================
def generate_report(metrics, y_true, y_pred, save_path):
    """Generate and save a comprehensive text report."""
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("DIABETIC RETINOPATHY CLASSIFICATION - PERFORMANCE REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("OVERALL PERFORMANCE:")
    report_lines.append("-" * 20)
    report_lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    report_lines.append(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    report_lines.append(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    report_lines.append(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    if metrics['roc_auc_macro'] is not None:
        report_lines.append(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    
    report_lines.append("")
    
    # Per-class metrics
    report_lines.append("PER-CLASS PERFORMANCE:")
    report_lines.append("-" * 25)
    for i, class_name in enumerate(CLASS_NAMES):
        report_lines.append(f"{class_name}:")
        report_lines.append(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        report_lines.append(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
        report_lines.append(f"  F1-Score: {metrics['f1_per_class'][i]:.4f}")
        if metrics['roc_auc_per_class'] is not None:
            report_lines.append(f"  ROC-AUC: {metrics['roc_auc_per_class'][i]:.4f}")
        report_lines.append("")
    
    # Classification report
    report_lines.append("DETAILED CLASSIFICATION REPORT:")
    report_lines.append("-" * 35)
    class_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    report_lines.append(class_report)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Detailed report saved to {save_path}")
    
    # Also print summary to console
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    if metrics['roc_auc_macro'] is not None:
        print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")

# ================================
# Main Execution
# ================================
def main():
    print("📊 Starting Performance Evaluation for Diabetic Retinopathy Detection")
    print("="*70)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    # Prepare test data
    test_image_paths = glob.glob(os.path.join(TEST_DIR, '**', '*.jpg'), recursive=True)
    
    if not test_image_paths:
        print(f"❌ No test images found in {TEST_DIR}")
        return
    
    print(f"📁 Found {len(test_image_paths)} test images")
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(test_image_paths, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Run inference
    y_true, y_pred, y_prob = run_inference(model, test_loader)
    
    # Calculate metrics
    print("📈 Calculating performance metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Generate visualizations
    print("🎨 Generating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # ROC Curves
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plot_roc_curves(y_true, y_prob, roc_path)
    
    # Class Distribution
    dist_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plot_class_distribution(y_true, y_pred, dist_path)
    
    # Per-class Metrics
    metrics_path = os.path.join(OUTPUT_DIR, "per_class_metrics.png")
    plot_per_class_metrics(metrics, metrics_path)
    
    # Generate Report
    report_path = os.path.join(OUTPUT_DIR, "performance_report.txt")
    generate_report(metrics, y_true, y_pred, report_path)
    
    print(f"\n✅ Evaluation complete! All results saved in '{OUTPUT_DIR}'")
    print("\n📋 Generated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png") 
    print("  - class_distribution.png")
    print("  - per_class_metrics.png")
    print("  - performance_report.txt")

if __name__ == "__main__":
    main()