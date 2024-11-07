# Loading ResNet18 from torchvision models
import torch
from torchvision.models import resnet18

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = resnet18(weights="DEFAULT")
model.to(device)
print(f"[INFO] ResNet18 Loaded to {device}")

dummy_input = torch.randn(64, 3, 224, 224)

# Analyze ResNet18 with OwLite wrapper
import owlite

owl = owlite.init(project="Introduction", baseline="resnet18")
# owl = owlite.init(project="Introduction", baseline="resnet18", experiment="first_quantization")

# Wrap model with owlite convert
model = owl.convert(model, dummy_input)

owl.export(model) # Export model to ONNX
owl.benchmark() # Benchmark model
