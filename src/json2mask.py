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
import tqdm
#Write the file path of the mask file to mask_dir
MASK_DIR  = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks'
#If there is no file in the given file path, a new file is created
if not os.path.exists(MASK_DIR): 
    os.mkdir(MASK_DIR)


jsons=os.listdir('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons')#List created with names of json files in ann folder


JSON_DIR = '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/jsons'#The path to the file is assigned to the variable
for json_name in tqdm.tqdm(jsons):#access the elements in the json list
    json_path = os.path.join(JSON_DIR, json_name)#Merged json_dir with json_name and created file path
    json_file = open(json_path, 'r')#file reading process
    json_dict=json.load(json_file)#Contents of json file converted to dict data type
    mask=np.zeros((json_dict["size"]["height"],json_dict["size"]["width"]), dtype=np.uint8)
    
    mask_path = os.path.join(MASK_DIR, json_name[:-5])
    # The values of the object keys in the dicts that we obtained from each 	json file have been added to the list.
    
    for obj in json_dict["objects"]:# To access each list inside the json_objs list
    
        if obj['classTitle']=='Freespace':#Objects whose classtitle is freespace
        
            cv2.fillPoly(mask,np.array([obj['points']['exterior']]),color=1)
            #Fill the masks created with np.zeros with the positions of the points in the json files.
            
    cv2.imwrite(mask_path,mask.astype(np.uint8))#Print filled masks in mask_path with imwrite
