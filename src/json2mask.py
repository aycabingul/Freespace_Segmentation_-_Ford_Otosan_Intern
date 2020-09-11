#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:08:11 2020

@author: aycaburcu
"""
import numpy as np
import cv2
import json
import os

#mask_dir'e mask dosyasının dosya yolunu yazdık
MASK_DIR  = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'
if not os.path.exists(MASK_DIR):#böyle bir dosya yolunda dosya yoksa 
    os.mkdir(MASK_DIR)#böyle bir dosya yolu olan dosya oluşturuyor


os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons')#geçerli çalışma dizinini verilen yola değiştirir
jsons=os.listdir()#ann klasörü içindeki json dosyalarının isimleriyle liste oluşturuldu


JSON_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons'#dosyanın yolu değişkene atandı
for json_name in jsons:#json listesinin içindeki elemanlara ulaşıldı
    json_path = os.path.join(JSON_DIR, json_name)#okunacak dosya yolu birleştirildi
    json_file = open(json_path, 'r')#dosya okuma işlemi
    json_dict=json.load(json_file)#json dosyasının içindekiler dict veri tipine çevrildi
    mask=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    # her bir json dosyasından elde ettiğimiz dict'lerin içerisindeki objects key'lerinin value'leri listeye eklendi
    
    for obj in json_dict["objects"]:# json_dict icindeki objects ulaşıldı 
    
        if obj['classTitle']=='Freespace':#classTitle==Freespace olanlar bulundu 
        
            cv2.fillPoly(mask,np.array([obj['points']['exterior']]),color=1)
            #np.zeros ile olusturdugumuz maskler json dosyalarımızın icersindeki point'lerin konumlarıyla dolduruldu 
            
    cv2.imwrite(mask_path,mask.astype(np.uint8))#imwrite ile mask_path içerisine doldurulan maskeler yazdırıldı