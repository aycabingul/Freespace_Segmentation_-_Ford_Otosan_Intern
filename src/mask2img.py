
# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
#kütüphaneler import edildi
import cv2 
import os
import numpy as np
import tqdm
from PIL import Image

#result yolunda bir dosya yoksa eğer o dosya oluşturuldu
result='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/result'
if not os.path.exists(result):
    os.mkdir(result)
#IMG_DIR yolundaki mask'e name'ler okunup bir listeye kaydedildi
IMG_DIR='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/images'
os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks')
masks_name=os.listdir()


os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/images')
image_name=os.listdir()

MASK_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'



for maskname in masks_name:#masks_name listesi'nin tek tek elemanlarına ulaşıldı
    img=cv2.imread(os.path.join(IMG_DIR,maskname)).astype(np.uint8)#maskname sonundan .png kesilerek .jpg eklenip okundu  
    mask=cv2.imread(os.path.join(MASK_DIR,maskname),0).astype(np.uint8)
    mask_ind   = mask == 1
    cpy_img  = img.copy()
    img[mask==1 ,:] = (255, 0, 125)
    opac_image=(img/2+cpy_img/2).astype(np.uint8)
    cv2.imwrite(os.path.join(result,maskname),opac_image)#yazdırıldı 
    
    
    
# new_image='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/new_image'
# for image in image_name:
#     img=cv2.imread(image)
#     img_name=image[:-3]+"png"
#     cv2.imwrite(os.path.join(new_image,img_name),img)