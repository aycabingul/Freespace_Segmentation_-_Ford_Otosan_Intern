import torch
import torch.nn as nn


def double_conv(in_channels,out_channels,mid_channels=None):#double_conv adında 2 parametresi olan bir function oluşturuldu
    if not mid_channels:
        mid_channels = out_channels
        return nn.Sequential(
        #nn.Sequential, o ağın yapı taşlarını (nn.Module'lar) sırayla belirterek bir sinir ağı oluşturmanıza olanak sağlar.
        
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),
        #conv2D, bir 2D veriye (ör. Bir görüntü) evrişim gerçekleştirme işlevidir.
        #Convolution, bir veri dizisi belirli bir işlemden geçtiğinde sonucu veren matematiksel bir operatördür.
        #Evrişimi görüntümüze bir filtre uygulamak olarak düşünün.
        #Conv2D işlevi, convolution'u çok daha karmaşık bir veri yapısına (sadece tek bir 2D veriye değil, 
        #bir 2D veri kümesine) ve padding ve stride gibi bazı ek seçeneklerle gerçekleştirebilir.
        #out_channels (int) - Evrişim tarafından üretilen kanal sayısı
        #kernel_size=2D convolution penceresinin genişliğini ve yüksekliğini belirten 2-tuple.
        #kernel_size (int veya tuple) - Konvolüsyon yapan çekirdeğin boyutu
        #padding (int veya tuple, isteğe bağlı) - Girişin her iki tarafına da sıfır dolgu eklendi (Varsayılan: 0)
        #in_channels 3 channels (renkli görüntüler) görüntüler için başlangıçta 3'tür. Siyah beyaz görüntüler için 1 olmalıdır. Bazı uydu görüntülerinde 4 olmalıdır.
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),#activation function
        #Tüm Convolutional katmanlarından sonra genellikle Non-Linearity(doğrusal olmayan) katmanı gellir.
        #Bu katman aktivasyon katmanı (Activation Layer) olarak adlandırılır çünkü aktivasyon fonksiyonlarından birini kullanılır.
        #aktivasyon fonksiyonu kullanılmayan bir neural network sınırlı öğrenme gücüne sahip bir linear regression gibi davranacaktır
        #ağınız çok derinse ve işlem yükünüz önemli bir problemse Relu seçilir
        #ReLU Fonksiyonu: Doğrultulmuş lineer birim (rectified linear unit- RELU) doğrusal olmayan bir fonksiyondur.
        #ReLU fonksiyonu negatif girdiler için 0 değerini alırken, x pozitif girdiler için x değerini almaktadır.
        #inplace=True herhangi bir ek çıktı tahsis etmeden girişi doğrudan değiştireceği anlamına gelir.
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
        #yapay sinir ağlarını yeniden ortalayarak ve yeniden ölçeklendirerek giriş katmanını normalleştirir
        #daha hızlı ve daha kararlı hale getirmek için kullanılan bir yöntemdir.
            nn.ReLU(inplace=True),
       
        
    )

class FoInternNet(nn.Module):#FoInternNet adında bir sınıf oluşturuldu 
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)  
        
        #self.dropout=nn.Dropout2d(0.5)
        self.maxpool = nn.MaxPool2d(2)#filtrenin gezdiği piksellerdeki değerlerin, maksimum olanını alır.
        #kernel_size=üzerinde “pool” yapılacak alanı belirler ve adım adım adımı belirler.
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        #Görüntü boyutlarını artırmak istediğimizde, temelde bir görüntüyü genişletiyoruz ve orijinal görüntünün satır ve sütunlarındaki "boşlukları" dolduruyoruz.
        #scale_factor:yukarı veya aşağı örneklemek için ölçek faktörü. Bir tuple ise, x ve y boyunca ölçeğe karşılık gelir.
        #Bilinear : Doğrusal enterpolasyonları kullanarak pikselin değerini hesaplamak için yakındaki tüm pikselleri kullanır.
        #align_corners=True, pikseller noktaları, bir kılavuz olarak kabul edilmektedir. Köşelerdeki noktalar hizalanır.
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        #print(x.shape)
        conv1 = self.dconv_down1(x)
    
        #print(conv1.shape)
        x = self.maxpool(conv1)
        
        #x=self.dropout(x)
        
        #print("maxpool")
        #print(x.shape)
        
        
        conv2 = self.dconv_down2(x)
    
        #print(conv2.shape)
        x = self.maxpool(conv2)
        #x=self.dropout(x)
        #print("maxpool")
        #print(x.shape)
        
        conv3 = self.dconv_down3(x)
        
        #print(conv3.shape)
        x = self.maxpool(conv3)   
        #x=self.dropout(x)
        #print("maxpool")
        #print(x.shape)
        
        x = self.dconv_down4(x)
        #print(x.shape)
        
        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)
        x = torch.cat([x, conv3], dim=1)#Verilen boyutta verilen tensör dizisini birleştirir 
        #dim: tensörlerin birleştirildiği boyut
     
    
        #print("cat")
        #print(x.shape)
        x = self.dconv_up3(x)

        #print(x.shape)

        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)

        x = torch.cat([x, conv2], dim=1)    
        #print("cat")
        #print(x.shape)


        x = self.dconv_up2(x)

        #print(x.shape)
        

        x = self.upsample(x)    
        #print("upsample")
        #print(x.shape)

        x = torch.cat([x, conv1], dim=1)   
        #print("cat")
        #print(x.shape)

        
        x = self.dconv_up1(x)

        #print(x.shape)

        
        x = self.conv_last(x)
        #print(x.shape)

        
        x = nn.Softmax(dim=1)(x)
        #print(x.shape)

        return x
    

