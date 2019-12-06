import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def resnet_block(input_layers, output_layers, k_size, stride, padding):
            

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ResnetBlockConv(nn.Module):
    def __init__(self, input_layers, output_layers, k_size, stride, padding):
        super(ResnetBlockConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class ResnetBlockConvolution
