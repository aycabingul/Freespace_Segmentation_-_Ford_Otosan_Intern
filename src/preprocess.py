import numpy as np
import cv2
import json
import os
import torch
import tqdm 


IMAGE_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/images"
MASK_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/masks"
batch_images=[]
image_path=[]
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
 

mask_path=[]
for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))

# img=cv2.imread(image_path[1],0)
# output_shape=list(img.shape[:2])
output_shape=[300,300]
def tensorize_image(image_path,output_shape):
    for image in tqdm.tqdm(image_path):
        img=cv2.imread(image)
        res=cv2.resize(img,tuple(output_shape))
        batch_images.append(res)
    tensor = torch.as_tensor(batch_images)
    #tensor = torch.Tensor([10,output_shape[0],output_shape[1],3])
    #return tensor

tensorize_image(image_path,output_shape)
batch_masks=[]
def tensorize_mask(mask_path,output_shape):
    for mask in tqdm.tqdm(mask_path):
        mask=cv2.imread(mask,0)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        res_mask = cv2.resize(mask, tuple(output_shape))
        n_classes=2
        one_hot = np.zeros((300, 300, n_classes))
        for i, unique_value in enumerate(np.unique(res_mask)):
            one_hot[:, :, i][res_mask==unique_value] = 1
        batch_masks.append(one_hot)
a=batch_masks[]

#tensore çevirmek için tensor=torch.Tensor(array) kullanıcaz