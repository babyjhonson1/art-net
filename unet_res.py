import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish = Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Остаточная связь
        out = self.mish(out)
        return out

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Кодировщик (32 -> 64 -> 128 -> 256)
        self.enc1 = ResidualBlock(1, 32)   
        self.pool1 = nn.MaxPool2d(2, 2)   
        self.enc2 = ResidualBlock(32, 64)  
        self.pool2 = nn.MaxPool2d(2, 2)   
        self.enc3 = ResidualBlock(64, 128) 
        self.pool3 = nn.MaxPool2d(2, 2)   
        self.enc4 = ResidualBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)   

        # Бутылочное горлышко
        self.bottleneck = ResidualBlock(256, 512) 

        # Декодировщик
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(512, 256) 
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(256, 128) 
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128, 64)  
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, 32)   
        
        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Кодировщик
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Бутылочное горлышко
        b = self.bottleneck(self.pool4(e4))

        # Декодировщик с остаточными связями
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.outconv(d1)
        return out