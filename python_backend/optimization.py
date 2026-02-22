import torch
import torch.nn as nn
import torch.quantization
from torch.nn.utils import prune
import copy

class ModelOptimizer:
    def __init__(self, model, input_shape=(1, 3, 512, 1024)):
        self.model = model
        self.input_shape = input_shape

    def apply_static_quantization(self, calibration_loader):
        """
        Apply Post-Training Static Quantization (PTQ) to reduce model size and latency.
        Converts FP32 weights/activations to INT8.
        """
        print("Applying Static Quantization...")
        self.model.eval()
        
        # Fuse Conv+BN+ReLU modules for efficiency
        # Note: This requires the model to have specific layer naming or structure
        # self.model.fuse_model() 
        
        # Configure quantization
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate with representative dataset
        with torch.no_grad():
            for images, _ in calibration_loader:
                self.model(images)
                
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        print("Quantization Complete.")
        return quantized_model

    def apply_pruning(self, amount=0.3):
        """
        Apply Unstructured Pruning (L1 Norm) to Conv2d and Linear layers.
        Removes 'amount' fraction of connections with smallest weights.
        """
        print(f"Pruning {amount*100}% of weights...")
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight') # Make pruning permanent
        print("Pruning Complete.")
        return self.model

    def knowledge_distillation(self, teacher_model, train_loader, optimizer, epochs=5, temp=7, alpha=0.3):
        """
        Train 'self.model' (Student) using 'teacher_model'.
        Loss = alpha * T * KL_Div + (1-alpha) * CrossEntropy
        """
        print("Starting Knowledge Distillation...")
        teacher_model.eval()
        self.model.train()
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                
                student_logits = self.model(images)
                
                # Calculate Loss
                loss_ce = criterion_ce(student_logits, labels)
                loss_kl = criterion_kl(
                    nn.functional.log_softmax(student_logits / temp, dim=1),
                    nn.functional.softmax(teacher_logits / temp, dim=1)
                ) * (temp * temp)
                
                loss = (1. - alpha) * loss_ce + alpha * loss_kl
                
                loss.backward()
                optimizer.step()
                
        print("Distillation Complete.")
        return self.model

    def export_to_onnx(self, path="model.onnx"):
        """
        Export model to ONNX format for cross-platform inference (TensorRT/OpenVINO).
        """
        print(f"Exporting to ONNX: {path}")
        dummy_input = torch.randn(self.input_shape)
        torch.onnx.export(
            self.model, 
            dummy_input, 
            path, 
            opset_version=13,
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Export Complete.")

# Example Usage
if __name__ == "__main__":
    from torchvision import models
    
    # 1. Load Teacher (ResNet152) and Student (MobileNetV3)
    teacher = models.resnet50(pretrained=True) # Using ResNet50 as proxy for 152
    student = models.mobilenet_v3_small(pretrained=True)
    
    optimizer = ModelOptimizer(student)
    
    # 2. Pruning
    student = optimizer.apply_pruning(amount=0.2)
    
    # 3. Export
    optimizer.export_to_onnx("ecg_mobilenet_optimized.onnx")
