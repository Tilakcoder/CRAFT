import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# (Re-use your _init_parameters and upconv_block functions)

def _init_parameters(net):
    for m in net:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

class upconv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upconv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel * 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel * 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# New extractor using ResNet50
class ResNetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetExtractor, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Build initial layers (conv1, bn1, relu, maxpool)
        self.initial = nn.Sequential(
            resnet.conv1,  # 7x7, stride 2 => output: (B, 64, H/2, W/2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # output: (B, 64, H/4, W/4) e.g., 128x128 for 512x512 input
        )
        # Extract features from each block:
        self.layer1 = resnet.layer1   # output: (B, 256, H/4, W/4)
        self.layer2 = resnet.layer2   # output: (B, 512, H/8, W/8)
        self.layer3 = resnet.layer3   # output: (B, 1024, H/16, W/16)
        self.layer4 = resnet.layer4   # output: (B, 2048, H/32, W/32)
        # Additional conv block to mimic conv_6 (reduce channels from 2048 to 512)
        self.conv_6 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        _init_parameters(self.conv_6.modules())
    
    def forward(self, x):
        x = self.initial(x)
        f1 = self.layer1(x)   # (B, 256, H/4, W/4) ~ 128x128 for 512x512 input
        f2 = self.layer2(f1)  # (B, 512, H/8, W/8) ~ 64x64
        f3 = self.layer3(f2)  # (B, 1024, H/16, W/16) ~ 32x32
        f4 = self.layer4(f3)  # (B, 2048, H/32, W/32) ~ 16x16
        f5 = self.conv_6(f4)  # (B, 512, H/32, W/32) ~ 16x16
        # Return in the order [f1, f2, f3, f4, f5] (from high to low resolution)
        return [f1, f2, f3, f4, f5]

# New merge module adjusted for ResNet feature dimensions
class merge_resnet(nn.Module):
    def __init__(self):
        super(merge_resnet, self).__init__()
        # f1: 256, f2: 512, f3: 1024, f4: 2048, f5: 512 (after conv_6)
        # We first merge f5 and f4 (channels 512+2048 = 2560) and reduce to 512.
        self.upconv1 = upconv_block(2560, 512)
        # Then merge with f3: 512 + 1024 = 1536, reduce to 256.
        self.upconv2 = upconv_block(1536, 256)
        # Then merge with f2: 256 + 512 = 768, reduce to 128.
        self.upconv3 = upconv_block(768, 128)
        # Then merge with f1: 128 + 256 = 384, reduce to 64.
        self.upconv4 = upconv_block(384, 64)
        
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.region_head = nn.Conv2d(16, 1, 1)
        self.affinity_head = nn.Conv2d(16, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        _init_parameters(self.modules())
                    
    def forward(self, feats):
        # feats: list = [f1, f2, f3, f4, f5]
        y = torch.cat((feats[4], feats[3]), 1)  # Concatenate f5 (512) and f4 (2048): 2560 channels
        y = self.upconv1(y)                    # => 512 channels
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        
        y = torch.cat((y, feats[2]), 1)          # 512 + 1024 = 1536 channels
        y = self.upconv2(y)                    # => 256 channels
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        
        y = torch.cat((y, feats[1]), 1)          # 256 + 512 = 768 channels
        y = self.upconv3(y)                    # => 128 channels
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        
        y = torch.cat((y, feats[0]), 1)          # 128 + 256 = 384 channels
        y = self.upconv4(y)                    # => 64 channels
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        
        y = self.conv(y)
        region_score = self.sigmoid1(self.region_head(y))
        affinity_score = self.sigmoid2(self.affinity_head(y))
        return region_score, affinity_score

# New CRAFT model using ResNet50 backbone
class CRAFT(nn.Module):
    def __init__(self, pretrained=True):
        super(CRAFT, self).__init__()
        self.extractor = ResNetExtractor(pretrained)
        self.merge = merge_resnet()

    def forward(self, x):
        feats = self.extractor(x)
        return self.merge(feats)

if __name__ == '__main__':
    # Test the new model with a dummy input
    model = CRAFT(pretrained=False)
    x = torch.randn(1, 3, 512, 512)
    region_score, affinity_score = model(x)
    print("Region score shape:", region_score.shape)
    print("Affinity score shape:", affinity_score.shape)
