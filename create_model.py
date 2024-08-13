from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur


class SimpleModel(nn.Module):
    """A simple example torch model containing only a gaussian blur"""

    def __init__(self):
        super().__init__()
        self.transform = GaussianBlur(kernel_size=3, sigma=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.transform(x)
            return y

class DConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, 1)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        return x2

class UNet(nn.Module):
    """A simple example torch model containing only a gaussian blur"""

    def __init__(self, in_chan, out_chan, fconv = 64):
        super().__init__()
        self.econv1 = DConv(in_chan, fconv)
        self.econv2 = DConv(fconv, fconv)
        self.bconv = DConv(fconv, fconv)
        self.dconv1 = DConv(fconv, fconv)
        self.dconv2 = DConv(fconv, fconv)
        self.out = nn.Conv2d(fconv, out_chan, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(2)

        self.transform = GaussianBlur(kernel_size=3, sigma=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.econv1(x)
        x2 = self.pool(self.econv2(x1))
        x3 = self.pool(self.bconv(x2))
        x4 = self.dconv1(torch.cat(self.up(x3), x2))
        x5 = self.dconv2(torch.cat(self.up(x4), x1))
        x6 = self.out(x5)
        return x6

def create_model(model_path: Path):
    """Create and save an example jit model"""
    # model = SimpleModel()
    model = UNet(1, 1, 64)
    example_input = torch.rand(1, 255, 255)
    jit_model = torch.jit.trace(model, example_inputs=example_input)
    print(f'Saving model to: {model_path.absolute()}')
    torch.jit.save(jit_model, model_path)


if __name__ == "__main__":
    model_path = Path(__file__).parent / "resources/model.pth"
    create_model(model_path)
