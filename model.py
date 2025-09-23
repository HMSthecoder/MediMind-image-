import torch
import torch.nn as nn
import timm

class DRModel(nn.Module):
    def __init__(self, num_classes=5):
        super(DRModel, self).__init__()
        # Load a pre-trained EfficientNet-B0 model
        self.base = timm.create_model("efficientnet_b0", pretrained=True)
        
        # Get the number of input features for the classifier
        in_features = self.base.classifier.in_features
        
        # Replace the original classifier with a new one for our specific number of classes
        self.base.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """The forward pass of the model."""
        return self.base(x)

if __name__ == '__main__':
    # This part is for testing the model structure. 
    # You don't need to run this file directly for training.
    model = DRModel(num_classes=5)
    print("Model created successfully!")
    print("\nModel Architecture:")
    print(model)
    
    # Create a dummy input tensor to test the forward pass
    dummy_input = torch.randn(2, 3, 224, 224) # (batch_size, channels, height, width)
    print(f"\nTesting forward pass with a dummy input of shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # The output shape should be (batch_size, num_classes), e.g., (2, 5)
    assert output.shape == (2, 5), "The output shape is incorrect!"
    
    print("\n✅ Model test passed!")
