from __future__ import print_function
import keras
import numpy as np
from pre_train_model_vgg19 import VGG_19
from plot import plot_matrix
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import Regularizer
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import math
import itertools


bird = glob.glob("./ph2_test/bird/*.JPEG")#bird
cat = glob.glob("./ph2_test/cat/*.JPEG")#cat
dog = glob.glob("./ph2_test/dog/*.JPEG")#dog
fish = glob.glob("./ph2_test/fish/*.JPEG")#fish
food = glob.glob("./ph2_test/food/*.JPEG")#food
insect = glob.glob("./ph2_test/insect/*.JPEG")#insect
plant = glob.glob("./ph2_test/plant/*.JPEG")#plant
rabbit = glob.glob("./ph2_test/rabbit/*.JPEG")#rabbit
scenery = glob.glob("./ph2_test/scenery/*.JPEG")#scenery
snake = glob.glob("./ph2_test/snake/*.JPEG")#snake

class_ = [dog,cat,bird,fish,insect,food,plant,rabbit,scenery,snake]
class_names = ['dog','cat','bird','fish','insect','food','plant','rabbit','scenery','snake']


#class_ = [cat,dog]
batch_size = 1
num_classes = 10
weight_filename = 'ph2_train_vgg_sp_cs.h5'
data_augmentation = True
threshold_moving = False

validation_data_dir = 'ph2_test'

load = True 

# input image dimensions
img_rows, img_cols = 128, 128

img_channels = 3

imput_size = (img_rows,img_cols,img_channels)

weight = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)




model = VGG_19(imput_size, include_top=False)
model.pop()
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu',kernel_initializer = weight, name='fc1_new'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',kernel_initializer = weight, name='predictions_new'))


##load 
if load == True :
    print('Model Load.')
    model.load_weights(weight_filename)




print(model.summary())


sgd = SGD(lr=0.0, decay=0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])




test_datagen = ImageDataGenerator(        
    featurewise_center=True,
    featurewise_std_normalization=True,)




#scoreSeg = model.evaluate_generator(validation_generator)
#print(scoreSeg)
#predict = model.predict_generator(validation_generator,1)
matrix = []

class_index = 0
for c in class_:     
     #print(class_index)
     class_list=[0,0,0,0,0,0,0,0,0,0]
     #class_list = [0,0]
     for path in c:
         img = image.load_img(path, target_size=(img_rows, img_cols))
         x = image.img_to_array(img)
         x = np.expand_dims(x, axis=0)
         predict = model.predict_proba(x)

         if threshold_moving == True:
            for i in range(len(class_)):
                predict[0][i] = predict[0][i] / len(class_[i])

         #print(predict)
         ans = np.where(predict==predict.max())
         predict_class = ans[1][0]
         class_list[predict_class] = class_list[predict_class] +1
         #print(predict_class)            
         
         #print(predict)
     matrix.append(class_list)
     print(matrix)
     class_index = class_index + 1
matrix = np.asarray(matrix)     
f = open('matrix.txt','w')
def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]

swap_cols(matrix, 0, 2)
swap_cols(matrix, 4, 5)
f.write(str(matrix))


performance_matrix = []
performance_matrix.append([])
performance_matrix.append([])
performance_matrix.append([])
performance_matrix.append([])

for i in range(num_classes):
    TP = 0
    FP = 0
    FN = 0
    for j in range(num_classes):
        for k in range(num_classes):
            if i == j and k == i :
                TP = matrix[j][k]
            elif j!= i and k == i:
                FP = FP + matrix[j][k]
            elif k != i and j == i:
                FN = FN + matrix[j][k]
    if TP+FP ==0:
        precision = 0
    else :
        precision = TP/(TP+FP)
        
    if TP+FN ==0:
        recall = 0
    else:
        recall = TP/(TP+FN)
    if (precision+recall) == 0:
        f_1 = 0
    else: 
        f_1 = 2*precision*recall/(precision+recall)
    if (precision*recall) == 0:
        g_measure = 0
    else :
        g_measure = math.sqrt(precision*recall)
    
    print('Precision : ',precision)
    print('Recall : ',recall)
    print('f_1 : ',f_1)
    print('g_measure : ',g_measure)
    performance_matrix[0].append(precision)
    performance_matrix[1].append(recall)
    performance_matrix[2].append(f_1)
    performance_matrix[3].append(g_measure)

plt.figure()

plot_matrix(matrix, classes=class_names,
                  title='Confusion matrix')
plt.savefig('confusion_matrix.png')

# draw performance matrix
plt.clf()
performance_matrix = np.asarray(performance_matrix)
for i in range(len(performance_matrix)):
    for j in range(len(performance_matrix[0])):
        performance_matrix[i][j] = round(performance_matrix[i][j], 2)

plot_matrix(performance_matrix, classes=class_names,
                  title='Performance matrix')
plt.savefig('performance_matrix.png')
# fig = plt.figure(figsize=(20, 5))
# plt.axis('off')
# plt.table(cellText=performance_matrix,colLabels=class_names,loc='center')

# plt.savefig('performance_matrix.png')
