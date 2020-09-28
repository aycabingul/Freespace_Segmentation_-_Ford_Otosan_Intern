import torch
import torch.nn as nn


def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1,padding=1),
        nn.ReLU(inplace=True)
        
    )

class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)  
        
        self.maxpool = nn.MaxPool2d(1)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)    

        x = torch.cat([x, conv2], dim=1)    


        x = self.dconv_up2(x)

        x = self.upsample(x)    

        x = torch.cat([x, conv1], dim=1)   

        
        x = self.dconv_up1(x)

        
        x = self.conv_last(x)

        
        x = nn.Softmax(dim=1)(x)

        return x
    

if __name__ == '__main__':
    model = FoInternNet(input_size=(224, 224), n_classes=2)