import torch
import torchvision

# efficientnet_v2_s
model = torchvision.models.efficientnet_b0(progress=True)
torch.save(model, "efficientnet-b0.pt")

