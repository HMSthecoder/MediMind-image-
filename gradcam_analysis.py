import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Import our custom modules
from model import DRModel
from augmentation import val_transform

# ================================
# Configuration
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"  # Your saved model
TEST_DIR = "data_processed/test"  # Test images directory
OUTPUT_DIR = "results/gradcam"  # Where to save Grad-CAM results
CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Load Model
# ================================
def load_trained_model():
    """Load the best trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Make sure you have trained the model first.")
        return None
    
    model = DRModel(num_classes=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    return model

# ================================
# Grad-CAM Functions
# ================================
def generate_gradcam(image_tensor, model, target_layers, class_idx=None):
    """Generate Grad-CAM heatmap for a given image and class."""
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    
    # Get the heatmap
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)[0, :]
    return grayscale_cam

def process_image_for_gradcam(image_path):
    """Load and preprocess image for Grad-CAM analysis."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize for display
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Apply the same transform as validation
    input_tensor = val_transform(image=image_resized)["image"].to(DEVICE)
    
    return image_normalized, input_tensor, image_resized

def visualize_gradcam(original_image, gradcam_heatmap, pred_class, true_class, confidence, save_path=None):
    """Create and display/save Grad-CAM visualization."""
    # Create the overlay
    visualization = show_cam_on_image(original_image, gradcam_heatmap, use_rgb=True)
    
    # Create subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image\nTrue Class: {CLASS_NAMES[true_class]}")
    axes[0].axis('off')
    
    # Grad-CAM overlay
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM Heatmap\nPredicted: {CLASS_NAMES[pred_class]} ({confidence:.2f}%)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

# ================================
# Main Analysis Functions
# ================================
def analyze_single_image(image_path, model, target_layers):
    """Analyze a single image with Grad-CAM."""
    # Extract true class from folder structure
    true_class = int(os.path.basename(os.path.dirname(image_path)))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n🔍 Analyzing: {image_path}")
    print(f"True class: {CLASS_NAMES[true_class]}")
    
    # Process image
    original_image, input_tensor, _ = process_image_for_gradcam(image_path)
    if original_image is None:
        print("❌ Failed to load image")
        return
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probabilities, 1)
        pred_class = pred_class.item()
        confidence = confidence.item() * 100
    
    print(f"Predicted class: {CLASS_NAMES[pred_class]} ({confidence:.2f}%)")
    
    # Generate Grad-CAM
    grayscale_cam = generate_gradcam(input_tensor, model, target_layers, class_idx=pred_class)
    
    # Create save path
    correctness = "correct" if pred_class == true_class else "wrong"
    save_path = os.path.join(OUTPUT_DIR, f"{image_name}_{correctness}_true{true_class}_pred{pred_class}.png")
    
    # Visualize
    visualize_gradcam(original_image, grayscale_cam, pred_class, true_class, confidence, save_path)
    
    return pred_class == true_class

def analyze_test_set(model, target_layers, max_images_per_class=3):
    """Analyze multiple images from the test set."""
    print(f"\n🎯 Analyzing test set with max {max_images_per_class} images per class...")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Process each class
    for class_idx in range(5):
        class_dir = os.path.join(TEST_DIR, str(class_idx))
        if not os.path.exists(class_dir):
            continue
        
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))[:max_images_per_class]
        
        print(f"\n📁 Class {class_idx} ({CLASS_NAMES[class_idx]}): {len(image_paths)} images")
        
        for image_path in image_paths:
            is_correct = analyze_single_image(image_path, model, target_layers)
            if is_correct is not None:
                total_predictions += 1
                if is_correct:
                    correct_predictions += 1
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n📊 Test Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    print("🔬 Starting Grad-CAM Analysis for Diabetic Retinopathy Detection")
    print("=" * 60)
    
    # Load model
    model = load_trained_model()
    if model is None:
        exit()
    
    # Define target layers for EfficientNet-B0
    # The conv_head is the last convolutional layer before classification
    target_layers = [model.base.conv_head]
    
    # Check if test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test directory '{TEST_DIR}' not found.")
        print("Make sure you have run preprocess.py to create the test set.")
        exit()
    
    # Analyze test set
    analyze_test_set(model, target_layers, max_images_per_class=2)
    
    print(f"\n✅ Analysis complete! Check '{OUTPUT_DIR}' for saved visualizations.")
    print("\n💡 Tips for your research paper:")
    print("- Use 'correct' predictions to show the model focuses on relevant lesions")
    print("- Use 'wrong' predictions to discuss model limitations and areas for improvement")
    print("- Compare heatmaps across different severity levels")