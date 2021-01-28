# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 02:25:33 2020

@author: aycaburcu
"""
#Let's import the libraries
import numpy as np
import cv2
import json
import os

#The path of the mask file has been written to mask_dir

MASK_DIR  = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)


os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons')#replaces the current working directory with the given path
jsons=os.listdir()#List created with names of json files in ann folder
a=0
json_objs=[]
json_dicts=[]

for i in jsons:#The elements in the json list have been reached
    JSON_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons'
    # file path assigned to variable
    json_name = i  #Every element in the jsons list is assigned to the variable
    json_path = os.path.join(JSON_DIR, json_name)
    #Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)
    #Contents of json file converted to dict data type
    json_dicts.append(json_dict)#each dict added to list
    json_objs.append(json_dict["objects"])
    # The values of the object keys in the dicts that we obtained from each 	json file have been added to the list.
point_list=[]
list_id=[]
two_fs=[]
for json_obj in json_objs:#To access each list inside the json_objs list
    for obj in json_obj:#to return the elements in the list
        obj_classtitle=obj['classTitle']
        if obj_classtitle=='Freespace':#Objects whose classtitle is freespace
            obj_points=obj['points']['exterior']
            #The points and exterior information of these objects were obtained.
            point_list.append(obj_points)#these are combined in a list
        else:
            continue
        #freespace id =38
        #The index of those with 38 id found in the list
        #these indexes are saved in the list
        for key, value in obj.items():
            if value==38:
                list_id.append(json_objs.index(json_obj))

#Since the freespaces of those with two freespaces were registered one after the other, the same indexes were printed one after the other. Indexes with the same index were found one after the other and they were assigned to a list
for i in range(4767):#There are 4768 elements in list_id, but since it starts from index 0, 4767 was received.
    if(list_id[i]==list_id[i+1]):
        two_fs.append(list_id[i])

two_point=[]
# find the first one in two freespaces, delete the first freespaces from the obj_point list and save the exteriors to another list
for fs in two_fs:
    two_point.append(point_list[fs])
    del point_list[fs]
two_fs
    
img_height = json_dict["size"]["height"]
img_width  = json_dict["size"]["width"]
img_size =[img_height,img_width]#image's height and width information assigned to a list

print("Toplam json dosya sayisi:", len(json_objs))
print("Goruntunun yuksekligi ve genisligi:", img_size)

#to get names of json files containing multiple freespaces
two_fs_files=[]
for i in two_fs:
    two_fs_files.append(jsons[i])
print("\nBirden fazla fs içeren dosyalar: ", two_fs_files)
two_fs
masks=(np.zeros(img_size, dtype=np.uint8))#create an empty mask

results=[]
for point in point_list:#To reach each element of the point_list
    mask=masks.copy()#copy blank mask to use for each image
    results.append(cv2.fillPoly(mask, np.array([point], dtype=np.int32), color=255))
    #fill the blank mask with the exteriors of each image

c=0
for point in two_point:
    #above it was writing on the blank mask
    mask=results[two_fs[c]]
    #here print a second freespace on the mask that was previously full of freespace and update the result
    results[two_fs[c]]=(cv2.fillPoly(mask, np.array([point], dtype=np.int32), color=255))
    c=c+1
os.chdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks')
#save each of the masks in the masks folder
v=0
for name in jsons:
    cv2.imwrite(name[:-5], results[v])
    #Convert each mask to png format by cutting the .json part at the end of the json file
    v=v+1
