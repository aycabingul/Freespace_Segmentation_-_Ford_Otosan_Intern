from matplotlib import pyplot as plt
from model1 import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check#Functions in preprocess file were imported
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.ticker as mticker
import torch
import cv2
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise

######### PARAMETERS ##########
valid_size = 0.3#Validation dataset is used to evaluate a particular model, but this is for frequent evaluation.
test_size  = 0.1#rate of data to be tested
batch_size = 8#it means how many data the model will process at the same time.
epochs = 25#Epoch count is the number of times all training data is shown to the network during training.
cuda =True
input_shape = (224, 224)#What size will the image resize
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()#The method tells us the location of the current working directory (CWD).
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
AUG_IMAGE=os.path.join(DATA_DIR,'augmentation')
AUG_MASK=os.path.join(DATA_DIR,'augmentation_mask')

###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
#The names of the files in the IMAGE_DIR path are listed and sorted
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# PREPARE IMAGE AND MASK LISTS
aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()






# DATA CHECK
image_mask_check(image_path_list, mask_path_list)
#Checked whether the elements in mask_path_list and image_path_list list are the same.



# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))
#A random array of permutations for the length of the image_path_list steps_per_epoch = len (train_input_path_list) // batch_size is created


# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)#Multiply indices length by test_size and assign it to an int-shaped variable
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind] #Get 0 to 476 elements of the image_path_list list
test_label_path_list = mask_path_list[:test_ind]#Get 0 to 476 elements of the mask_path_list list

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the image_path_list list
valid_label_path_list = mask_path_list[test_ind:valid_ind]#Get 476 to 1905 elements of the mask_path_list list

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]#Get the elements of the image_path_list list from 1905 to the last element
train_label_path_list = mask_path_list[valid_ind:]#Get the elements of the mask_path_list list from 1905 to the last element
#burada yukarıda vermiş olduğumuz test verisi için tüm datanın 0.1 ve validation verisi tüm datanın 0.3 içermeli
#Here, for the test data we have given above, all the data should contain 0.1 and all the validation data should contain 0.3, 
#but both of them do not belong to the same data data.

# train_input_path_list.extend(aug_path_list)
# train_label_path_list.extend(aug_mask_path_list)



aug_size=int(len(aug_mask_path_list)/2)
train_input_path_list=aug_path_list[:aug_size]+train_input_path_list+aug_path_list[aug_size:]
train_label_path_list=aug_mask_path_list[:aug_size]+train_label_path_list+aug_mask_path_list[aug_size:]
#One time rounding (updating of weights) of the entire data set through neural networks is called an epoch.


steps_per_epoch = len(train_input_path_list)//batch_size
# Find how many times to do it by dividing the length of the train data (training data) by batch_size
#in an epoch, a data string in the dataset goes to the end in neural networks
#It then waits there until the batch reaches you, the error rate is calculated after the data reaches the end
#Divide the training data set by 4 since our batch_size is 4

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=2)
#Enter parameters into model and assign output to variable

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()#Creates a criterion that measures the Binary Cross Entropy between target and output:
#BCELoss is an acronym for Binary CrossEntropyLoss, a special case of BCOMoss CrossEntropyLoss used only for two categories of problems.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#Commonly used momentum beta coefficient is 0.9.
#lr=learning rate

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

val_losses=[]
train_losses=[]
# TRAINING THE NEURAL NETWORK
for epoch in tqdm.tqdm(range(epochs)):

    running_loss = 0
    #In each epoch, images and masks are mixed randomly in order not to output images sequentially.
    pair_IM=list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object=zip(*pair_IM)
    zipped_list=list(unzipped_object)
    train_input_path_list=list(zipped_list[0])
    train_label_path_list=list(zipped_list[1])
    
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        #train_input_path_list [0: 4] gets first 4 elements on first entry
        #in the second loop train_input_list [4: 8] gets the second 4 elements
        #element advances each time until batch_size
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        #Our data that we will insert into the model in the preprocess section is prepared by entering the parameters.
        
        optimizer.zero_grad()#gresets the radian otherwise accumulation occurs on each iteration
        # Manually reset gradients after updating Weights

        outputs = model(batch_input) # Give the model batch_input as a parameter and assign the resulting output to the variable.
        

        # Forward passes the input data
        loss = criterion(outputs, batch_label)
        loss.backward()# Calculates the gradient, how much each parameter needs to be updated
        optimizer.step()# Updates each parameter according to the gradient

        running_loss += loss.item()# loss.item () takes the scalar value held in loss.

        print(ind)
        #validation 
        if ind == steps_per_epoch-1:
            
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                val_losses.append(val_loss)
                break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            
torch.save(outputs, '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/src/best_model.pth')
print("Model Saved!")
best_model = torch.load('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/src/best_model.pth')

def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
    
    plt.show()

draw_graph(val_losses,train_losses,epochs)


def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==1 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('image', 'predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)


    

# zip:
#letters = ['a', 'b', 'c']
#numbers = [0, 1, 2]
#for l, n in zip(letters, numbers):
    #print(f'Letter: {l}')
    #print(f'Number: {n}')
# Letter: a
# Number: 0
# Letter: b
# Number: 1
# Letter: c
# Number: 2