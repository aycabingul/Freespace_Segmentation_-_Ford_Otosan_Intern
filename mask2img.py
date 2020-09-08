# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 
import os
import numpy as np

result='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/result'
if not os.path.exists(result):
    os.mkdir(result)
    
IMG_DIR='/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/images'
os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks')#geçerli çalışma dizinini verilen yola değiştirir
masks_name=os.listdir()#ann klasörü içindeki json dosyalarının isimleriyle liste oluşturuldu


MASK_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'


for maskname in masks_name:
    img=cv2.imread(os.path.join(IMG_DIR,maskname[:-4]+'.jpg'))
    mask=cv2.imread(os.path.join(MASK_DIR,maskname),0)
    res=cv2.bitwise_or(img,img,mask=mask)
    cv2.imwrite(os.path.join(result,maskname[:-4]+'.jpg'),res)
