import torch
import torchvision

model = torchvision.models.resnet50(progress=True)
torch.save(model, "resnet50.pt")

model = torchvision.models.quantization.resnet50(quantize=True, progress=True)
torch.save(model, "resnet50-int8.pt")
