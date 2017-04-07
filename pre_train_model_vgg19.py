from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD


def VGG_19(input_shape=None,include_top=True):

    model = Sequential()

    model.add(Conv2D(64, (3, 3),input_shape=input_shape, padding='same', activation='relu', name='block1_conv1'))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block1_pool'))


    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))
  
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block2_pool'))


    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))
  
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block3_pool'))


    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block4_pool'))


    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1'))
 
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2'))
 
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv4'))
    model.add(MaxPooling2D((2,2), strides=(2,2), name='block5_pool'))

    if include_top:
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='fc1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc2'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax', name='predictions'))
        

    #print(model.summary())
  

    return model
