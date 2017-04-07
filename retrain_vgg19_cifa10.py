from __future__ import print_function
from pre_train_model_vgg19 import VGG_19
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import Regularizer
import tensorflow as tf
import numpy as np
import math
import glob

load = True 
batch_size = 128
num_classes = 10
epochs = 10
weight_filename = 'ph2_train_vgg_sp_cs.h5'
#train_data_dir = 'ph2_train'
train_data_dir = 'ph2_train_sp'  #sample

#validation_data_dir = 'ph2_test'
#nb_validation_samples = 2978


# input image dimensions
img_rows, img_cols = 128, 128
img_channels = 3

imput_size = (img_rows,img_cols,img_channels)


bird = glob.glob("./"+train_data_dir+"/bird/*.JPEG")#bird
cat = glob.glob("./"+train_data_dir+"/cat/*.JPEG")#cat
dog = glob.glob("./"+train_data_dir+"/dog/*.JPEG")#dog
fish = glob.glob("./"+train_data_dir+"/fish/*.JPEG")#fish
food = glob.glob("./"+train_data_dir+"/food/*.JPEG")#food
insect = glob.glob("./"+train_data_dir+"/insect/*.JPEG")#insect
plant = glob.glob("./"+train_data_dir+"/plant/*.JPEG")#plant
rabbit = glob.glob("./"+train_data_dir+"/rabbit/*.JPEG")#rabbit
scenery = glob.glob("./"+train_data_dir+"/scenery/*.JPEG")#scenery
snake = glob.glob("./"+train_data_dir+"/snake/*.JPEG")#snake
count_class = [len(bird),len(cat),len(dog),len(fish),len(food),len(insect),len(plant),len(rabbit),len(scenery),len(snake)]
print(count_class)
nb_train_samples = len(bird)+len(cat)+len(dog)+len(fish)+len(food)+len(insect)+len(plant)+len(rabbit)+len(scenery)+len(snake)




weight = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
##build vgg19 model
model = VGG_19(imput_size, include_top=False)
model.pop()
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu',kernel_initializer = weight, name='fc1_new'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',kernel_initializer = weight, name='predictions_new'))

if load == True:
    model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

print(model.summary())


def step_decay(epoch):
    initial_lrate = 0.0001
    lrate = initial_lrate
    
    lrate = lrate / (0.5*epoch)

    print(lrate,epoch)
    return lrate
    
    
sgd = SGD(lr=0.0001, decay=0, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5)
rms =keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Let's train the model using SGD with momentum
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])



train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    shear_range=0.125,
    zoom_range=0.125,
    horizontal_flip=True)



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    #color_mode='grayscale',
    class_mode='categorical',
    shuffle = True)



# labels_dict : {ind_label: count_label}
# mu : parameter to tune 
def create_class_weight(labels_dict,mu=1):
    
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*nb_train_samples/float(labels_dict[key]))
        class_weight[key] = score #if score > 1.0 else 1.0

    return class_weight

# random labels_dict
labels_dict = {0: count_class[0], 1: count_class[1], 2: count_class[2], 3: count_class[3],
 4: count_class[4], 5: count_class[5], 6: count_class[6], 7: count_class[7], 8:count_class[8], 9:count_class[9]}


class_w=create_class_weight(labels_dict)
print(class_w)
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    class_weight=class_w
#validation_data=validation_generator,
#validation_steps=nb_validation_samples // batch_size
)
        



model.save_weights(weight_filename)
