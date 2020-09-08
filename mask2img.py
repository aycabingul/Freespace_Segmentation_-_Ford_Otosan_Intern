# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#kütüphaneler import edildi
import cv2 
import os
import numpy as np

#result yolunda bir dosya yoksa eğer o dosya oluşturuldu
result='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/result'
if not os.path.exists(result):
    os.mkdir(result)
#IMG_DIR yolundaki mask'e name'ler okunup bir listeye kaydedildi
IMG_DIR='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/images'
os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks')
masks_name=os.listdir()


MASK_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'


for maskname in masks_name:#masks_name listesi'nin tek tek elemanlarına ulaşıldı
    img=cv2.imread(os.path.join(IMG_DIR,maskname[:-4]+'.jpg'))#maskname sonundan .png kesilerek .jpg eklenip okundu  
    mask=cv2.imread(os.path.join(MASK_DIR,maskname),0)
    res=cv2.bitwise_and(img,img,mask=mask)#mask'ın üzerine resim yazdırıldı 
    cv2.imwrite(os.path.join(result,maskname[:-4]+'.jpg'),res)#yazdırıldı 
