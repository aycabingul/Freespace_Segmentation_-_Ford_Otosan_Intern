import torch
import torch.nn as nn


def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1),
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
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        print(x.shape)
        conv1 = self.dconv_down1(x)
        print(conv1.shape)
        x = self.maxpool(conv1)
        print("maxpool")
        print(x.shape)
        
        
        conv2 = self.dconv_down2(x)
        print(conv2.shape)
        x = self.maxpool(conv2)
        print("maxpool")
        print(x.shape)
        
        conv3 = self.dconv_down3(x)
        print(conv3.shape)
        x = self.maxpool(conv3)   
        print("maxpool")
        print(x.shape)
        
        x = self.dconv_down4(x)
        print(x.shape)
        
        x = self.upsample(x)    
        print("upsample")
        print(x.shape)
        x = torch.cat([x, conv3], dim=1)
        print("cat")
        print(x.shape)
        x = self.dconv_up3(x)
        print(x.shape)

        x = self.upsample(x)    
        print("upsample")
        print(x.shape)

        x = torch.cat([x, conv2], dim=1)    
        print("cat")
        print(x.shape)


        x = self.dconv_up2(x)
        print(x.shape)
        

        x = self.upsample(x)    
        print("upsample")
        print(x.shape)

        x = torch.cat([x, conv1], dim=1)   
        print("cat")
        print(x.shape)

        
        x = self.dconv_up1(x)
        print(x.shape)

        
        x = self.conv_last(x)
        print(x.shape)

        
        x = nn.Softmax(dim=1)(x)
        print(x.shape)

        return x
    

if __name__ == '__main__':
    model = FoInternNet(input_size=(224, 224), n_classes=2)