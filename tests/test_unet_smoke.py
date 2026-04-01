import torch
from src.model.unet import UNet

if __name__ == '__main__':
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    print('input shape:', x.shape)
    print('output shape:', y.shape)
