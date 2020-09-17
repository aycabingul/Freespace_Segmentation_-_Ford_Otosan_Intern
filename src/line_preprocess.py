import numpy as np
import cv2
import json
import os
import torch
import tqdm 


IMAGE_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/image"
#image klasörünün yolu değişkene atandı 
MASK_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/maskline"
#masks klasörünün yolu değişkene atandı
batch_images=[] #boş liste oluşturuldu
image_path=[] #boş liste oluşturuldu
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
#IMAGE_DIR yolundaki klasörün içindeki dosyaların isimleri bir listeye kaydedildi 

mask_path=[] #boş bir liste oluşturuldu
for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
#MASK_DIR yolundaki klasörün içindeki dosyaların isimleri bir yoluyla birleştirilerek bir listeye kaydedildi
output_shape=[224,224]
batch_masks=[]#boş liste oluşturuldu


def tensorize_image(image_path,output_shape): #2 parametreli fonksiyon oluşturuldu
    global tensor_image
    for image in tqdm.tqdm(image_path[:4]): #for döngüsü ile image_path listesinin içindeki elemanlara tek tek ulaşıldı
        img=cv2.imread(image) #image değişkenine atanmış dosya yolundaki,dosya okundu
        res=cv2.resize(img,tuple(output_shape)) #image'a resize işlemi uygulandı 
        batch_images.append(res) #resize değiştirilmiş resimler listeye kaydedildi
    image_torch=torch.as_tensor(batch_images,dtype=torch.float32)#yukarıda oluşturulan liste torch tensor'e çevrildi
    tensor_image=image_torch.cuda()
    print(tensor_image.size())
    #[4765,300,300,3] 
    return tensor_image #tensor return edildi




def tensorize_mask(mask_path,output_shape,n_classes):#iki parametreye sahip function oluşturuldu
    global tensor_mask
    for mask in tqdm.tqdm(mask_path[:4]):#mask_path listesinin elemanlarına tek tek ulaşıldı
        mask=cv2.imread(mask,0)#dosyalar okundu 
        #buradaki bir değişiklik (HXW) şeklinde okundu(siyah ,beyaz)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    
        res_mask = cv2.resize(mask, tuple(output_shape))#image'a resize işlemi uygulandı 
        #one hot encode 
        mask=one_hot_encoder(res_mask,n_classes)
        batch_masks.append(mask)
        mask_torch=torch.as_tensor(batch_masks)
        tensor_mask=mask_torch.cuda()
    print(tensor_mask.size())
    #[4765,300,300,2] 
    return tensor_mask#tensor return  edildi 

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



batch_image_tensor=tensorize_image(image_path,output_shape)#fonksiyon çağrıldı

batch_mask_tensor=tensorize_mask(mask_path,output_shape,3)#function çağrıldı 

print(batch_image_tensor.shape)
print(batch_image_tensor.dtype)
print(batch_image_tensor.type)

print(batch_mask_tensor.shape)
print(batch_mask_tensor.dtype)
print(batch_mask_tensor.type)




