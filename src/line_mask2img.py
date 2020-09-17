
#kütüphaneler import edildi
import cv2 
import os
import numpy as np
import tqdm

#result yolunda bir dosya yoksa eğer o dosya oluşturuldu
result='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/resultline'
if not os.path.exists(result):
    os.mkdir(result)
IMG_DIR='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/image'
masks_name=os.listdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/maskline')

MASK_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/maskline'

#png formatındaki img'ler png formatına çevrildi
# for maskname in tqdm.tqdm(masks_name):
#     img=cv2.imread(os.path.join(IMG_DIR,maskname[:-4]+".jpg")).astype(np.uint8)
#     cv2.imwrite(os.path.join(IMG_DIR,maskname),img)
#     os.remove(os.path.join(IMG_DIR,maskname[:-4]+".jpg"))#png ile kaydedildikten sonra jpg olanlar silindi

for maskname in tqdm.tqdm(masks_name):#masks_name listesi'nin tek tek elemanlarına ulaşıldı

    img=cv2.imread(os.path.join(IMG_DIR,maskname)).astype(np.uint8)
    mask=cv2.imread(os.path.join(MASK_DIR,maskname),0).astype(np.uint8)
    mask_ind   = mask == 1

    cpy_img  = img.copy()

    img[mask==1 ,:] = (0, 0, 255)
    img[mask==2,:]=(38, 255, 255)

    opac_image=(img/2+cpy_img/2).astype(np.uint8)
    cv2.imwrite(os.path.join(result,maskname),opac_image)#yazdırıldı 
 
    



