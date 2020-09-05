# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 02:25:33 2020

@author: aycaburcu
"""
#kütüphaneleri import ettik 
import numpy as np
import pandas as pd
import cv2
import json
import os

#mask_dir'e mask dosyasının dosya yolunu yazdık
MASK_DIR  = 'D:/Belgeler/ders/jsonandmask/L4 Images/2020_02_22_FC120/masks'
if not os.path.exists(MASK_DIR):#böyle bir dosya yolunda dosya yoksa 
    os.mkdir(MASK_DIR)#böyle bir dosya yolu olan dosya oluşturuyor


os.chdir('D:/Belgeler/ders/jsonandmask/L4 Images/2020_02_22_FC120/ann')#geçerli çalışma dizinini verilen yola değiştirir
jsons=os.listdir()#ann klasörü içindeki json dosyalarının isimleriyle liste oluşturuldu

a=0
json_objs=[]
json_dicts=[]

for i in jsons:#json listesinin içindeki elemanlara ulaşıldı
    JSON_DIR = 'D:/Belgeler/ders/jsonandmask/L4 Images/2020_02_22_FC120/ann'#dosyanın yolu değişkene atandı
    json_name = i#jsons listesindeki her eleman değişkene atandı
    json_path = os.path.join(JSON_DIR, json_name)#okunacak dosya yolu birleştirildi
    json_file = open(json_path, 'r')#dosya okuma işlemi
    json_dict=json.load(json_file)#json dosyasının içindekiler dict veri tipine çevrildi
    json_dicts.append(json_dict)#her dict listeye eklendi
    json_objs.append(json_dict["objects"])
    # her bir json dosyasından elde ettiğimiz dict'lerin içerisindeki objects key'lerinin value'leri listeye eklendi
point_list=[]
list_id=[]
two_fs=[]
for json_obj in json_objs:#json_objs listesinin içindeki her listeye ulaşmak için
    for obj in json_obj:#listenin içindeki elemanları dönmek için
        obj_classtitle=obj['classTitle']
        if obj_classtitle=='Freespace':#classtitle freespace olanları bulmak için
            obj_points=obj['points']['exterior']#class title freespace olanların points'deki exteriorlarını aldık
            point_list.append(obj_points)#bunları bir listede birlestirdik
        else:
            continue
        #freespace'in id'si =38
        #id'si 38 olanların listede hangi indexde olduğunu buldum
        #bu indexleri listeye kaydettim
        for key, value in obj.items():
            if value==38:
                list_id.append(json_objs.index(json_obj))

#iki tane freespace olanların freespace'leri peş peşe kaydedildiği için 
#peş peşe aynı indexleri yazdırmış olduk
#peş peşe aynı index olan indexleri buldum 
#bunları bir listeye atadım
for i in range(4768):
    if(list_id[i]==list_id[i+1]):
        two_fs.append(list_id[i])

two_point=[]
# iki tane freespace bulup, ilk freespace'leri obj_point listesinden silip
#başka bir listeye exterior'larını kaydettim
for fs in two_fs:
    a=0
    two_point.append(point_list[fs+a])
    del point_list[fs+a]
    a=a+1
#ilk başta bizim listemizde freespace sırayla yazılıyordu ama iki tane freespace olduğu zaman 
#liste her seferinde bir index ileri haydığı için fs+s aldım
#ilk iki freespace ait index 91 ozaman ikinci freespace 92 yazılmıştır
#ama 3011 deki freespaceler ise 3012 ve 3013 yazılmıştır 


    
img_height = json_dict["size"]["height"]
img_width  = json_dict["size"]["width"]
img_size =[img_height,img_width]#image'in height ve width bir listeye atadık

print("Toplam json dosya sayisi:", len(json_objs))
print("Goruntunun yuksekligi ve genisligi:", img_size)
print(two_fs)

two_fs_files=[]
for i in two_fs:
    two_fs_files.append(jsons[i])
print("\nBirden fazla fs içeren dosyalar: ", two_fs_files)
masks=(np.zeros(img_size, dtype=np.uint8))#boş bir maske oluşturduk 

results=[]
for point in point_list:#point_liste'sinin her bir elemanına ulaşmak için 
    mask=masks.copy()#boş maskeyi her image için kullanmak için kopyaladık
    results.append(cv2.fillPoly(mask, np.array([point], dtype=np.int32), color=255))
    #boş maskeyi her image'ın exteriorları ile doldurduk 

c=0
for point in two_point:
    #üstte boş maskenin üzerine yazıyorduk normalde
    mask=results[two_fs[c]]#burada daha önceden freespace dolu olan maske'nin üzerine
    #ikinci freespace yazdırıp
    # result'ı güncelledim
    results[two_fs[c]]=(cv2.fillPoly(mask, np.array([point], dtype=np.int32), color=255))
    c=c+1
os.chdir('D:/Belgeler/ders/jsonandmask/L4 Images/2020_02_22_FC120/masks')#geçerli çalışma dizinini verilen yola değiştirir
#mask'lerin her birini masks klasörüne kaydetmek için yapıldı
v=0
for name in jsons:#her ann dosyasındaki json  dosyalarının  isimlerine ulaşmak için
    cv2.imwrite(name[:-5], results[v])
    #json dosyasının sonundaki .json kısmını keserek her maske'yi png formatına çevirdik
    v=v+1
