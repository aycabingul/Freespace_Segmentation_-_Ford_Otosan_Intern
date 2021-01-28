import numpy as np
import cv2
import json
import os
import torch
import tqdm 


IMAGE_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/image"
#The path to the image folder is assigned to the variable
MASK_DIR="/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/data/maskline"
#The path to the mask folder is assigned to the variable
batch_images=[] 
image_path=[] 
for name in os.listdir(IMAGE_DIR):
    image_path.append(os.path.join(IMAGE_DIR,name))
#Save the names of the files in the folder in IMAGE_DIR path to the list

mask_path=[] 
for name in os.listdir(MASK_DIR):
    mask_path.append(os.path.join(MASK_DIR,name))
#Save the names of the files in the folder in MASK_DIR path to the list
output_shape=[224,224]
batch_masks=[]


def tensorize_image(image_path,output_shape): #Create a 2-parameter function
    global tensor_image
    for image in tqdm.tqdm(image_path[:4]): #Access the elements in the image_path list one by one with the for loop
        img=cv2.imread(image) #Read file in path assigned to image variable
        res=cv2.resize(img,tuple(output_shape)) #Apply image resize operation
        batch_images.append(res) #save resized images to list
    image_torch=torch.as_tensor(batch_images,dtype=torch.float32)#convert the list to torch tensor
    tensor_image=image_torch.cuda()
    print(tensor_image.size())
    #[4765,300,300,3] 
    return tensor_image 




def tensorize_mask(mask_path,output_shape,n_classes):#Create a 2-parameter function
    global tensor_image
    global tensor_mask
    for mask in tqdm.tqdm(mask_path[:4]):#Access the elements in the mask_path list one by one with the for loop
        mask=cv2.imread(mask,0)#read files
        #a change here; Read as (HXW) (black, white)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)    
        res_mask = cv2.resize(mask, tuple(output_shape))#Apply image resize operation
        #one hot encode 
        mask=one_hot_encoder(res_mask,n_classes)
        batch_masks.append(mask)
        mask_torch=torch.as_tensor(batch_masks)
        tensor_mask=mask_torch.cuda()
    print(tensor_mask.size())
    #[4765,300,300,2] 
    return tensor_mask

def one_hot_encoder(res_mask,n_classes):
    
    #one hot encode
    #Create an np.array of zeros.
    one_hot=np.zeros((res_mask.shape[0],res_mask.shape[1],n_classes),dtype=np.int)
    #Find unique values in res_mask [0,1,2]
    #increase in i by the length of the list
    #[0,1,2] when returning the inside of list, each list element is given to unique_value variable
    for i,unique_value in enumerate(np.unique(res_mask)):
        one_hot[:,:,i][res_mask==unique_value]=1
    return one_hot



batch_image_tensor=tensorize_image(image_path,output_shape)

batch_mask_tensor=tensorize_mask(mask_path,output_shape,3) 

print(batch_image_tensor.shape)
print(batch_image_tensor.dtype)
print(batch_image_tensor.type)

print(batch_mask_tensor.shape)
print(batch_mask_tensor.dtype)
print(batch_mask_tensor.type)




