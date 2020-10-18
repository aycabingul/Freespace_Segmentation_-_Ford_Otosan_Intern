import numpy as np
import cv2
import json
import os
import torch
import tqdm 
from matplotlib import pyplot as plt




def tensorize_image(image_path,output_shape,cuda=False): #2 parametreli fonksiyon oluşturuldu
    batch_images=[] #boş liste oluşturuldu
    for image in image_path[:8]: #for döngüsü ile image_path listesinin içindeki elemanlara tek tek ulaşıldı
        img=cv2.imread(image) #image değişkenine atanmış dosya yolundaki,dosya okundu
        norm_img = np.zeros((1920,1208))
        final_img = cv2.normalize(img,  norm_img, 0, 255, cv2.NORM_MINMAX)
        img=cv2.resize(final_img,tuple(output_shape)) #image'a resize işlemi uygulandı 
        torchlike_image = torchlike_data(img)
        batch_images.append(torchlike_image) #resize değiştirilmiş resimler listeye kaydedildi
    torch_image=torch.as_tensor(batch_images,dtype=torch.float32).float()#yukarıda oluşturulan liste torch tensor'e çevrildi
    
    
    if cuda:
        torch_image= torch_image.cuda()
    return torch_image
    print(torch_image.size())
    #[4765,300,300,3] 




def tensorize_mask(mask_path,output_shape,n_classes,cuda=False):#iki parametreye sahip function oluşturuldu
    batch_masks=[]
    for mask in mask_path:#mask_path listesinin elemanlarına tek tek ulaşıldı
        mask=cv2.imread(mask,0)#dosyalar okundu 
        #buradaki bir değişiklik (HXW) şeklinde okundu(siyah ,beyaz)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    
        mask= cv2.resize(mask, tuple(output_shape))#image'a resize işlemi uygulandı 
        #one hot encode 
        mask=one_hot_encoder(mask,n_classes)
        torchlike_mask = torchlike_data(mask)
        batch_masks.append(torchlike_mask)
    torch_mask=torch.as_tensor(batch_masks,dtype=torch.float32)

    
    if cuda:
        torch_mask=torch_mask.cuda()
    return torch_mask
    #[4765,300,300,2] 


def one_hot_encoder(res_mask,n_classes):
    #one hot encode
    #birtane np arrayi oluşturuyoruz sıfırlardan oluşan (224,224,2)'lik
    one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
    #res_mask içerisindeki eşsiz değerleri bulduk[0,1]
    #i'de liste'nin uzunluğu kadar artıcak 
    #[0,1] listesi'nin içini dönerken her liste elemanı unique_value değişkenine veriliyor
    for i,unique_value in enumerate(np.unique(res_mask)):
        one_hot[:,:,i][res_mask==unique_value]=1
    return one_hot

def torchlike_data(data):
    #transpose işlemi 
    n_channels = data.shape[2]
    torchlike_data = np.empty((n_channels, data.shape[0], data.shape[1]))#verilen şekil ve türde yeni bir dizi döndürür.
    #bu ölçülerde bir dizi oluşturuyor 
    for ch in range(n_channels):#liste'nin uzunluğu kadar ch sayısı üretir
        torchlike_data[ch] = data[:,:,ch]#torchlike_data[0]=data[:,:,0] şeklinde eşitlenir
        #data'nın içindeki veriler tek tek torchlike_data aktarılır
    return torchlike_data

#mask ismi ile image ismi uyuyormu ona bakıyor uymuyorsa uyarı çıkarıyor ekrana 
def image_mask_check(image_path_list, mask_path_list):
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name  = mask_path.split('/')[-1].split('.')[0]
        assert image_name == mask_name, "Image and mask name does not match {} - {}".format(image_name, mask_name)


if __name__ == '__main__':
    output_shape=[224,224]
    IMAGE_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/image"
    #image klasörünün yolu değişkene atandı 
    MASK_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks"
    #masks klasörünün yolu değişkene atandı
    image_path=[] #boş liste oluşturuldu
    for name in os.listdir(IMAGE_DIR):
        image_path.append(os.path.join(IMAGE_DIR,name))
    batch_image_list=image_path[:4]
    batch_image_tensor=tensorize_image(batch_image_list,output_shape)#fonksiyon çağrıldı
    print(batch_image_list)
    print(batch_image_tensor.dtype)
    print(type(batch_image_tensor))
    print(batch_image_tensor.shape)

    
    #IMAGE_DIR yolundaki klasörün içindeki dosyaların isimleri bir listeye kaydedildi 
    
    mask_path=[] #boş bir liste oluşturuldu
    for name in os.listdir(MASK_DIR):
        mask_path.append(os.path.join(MASK_DIR,name))
    batch_mask_list=mask_path[:4]
    batch_mask_tensor=tensorize_mask(batch_mask_list,output_shape,2)#function çağrıldı 
    #MASK_DIR yolundaki klasörün içindeki dosyaların isimleri bir yoluyla birleştirilerek bir listeye kaydedildi
    print(batch_mask_list)
    print(batch_mask_tensor.dtype)
    print(type(batch_mask_tensor))
    print(batch_mask_tensor.shape)  


