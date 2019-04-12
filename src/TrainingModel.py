
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import warnings
warnings.filterwarnings("ignore")

# dimensions of our images
img_width, img_height = 250, 250

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 364
nb_validation_samples = 52
epochs = 50
batch_size = 30
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
# instanstiate CNN
model = Sequential()

# Modulating enthropic capacity, by adding 3 conv2d layers and max pooling layers 
# This is very similar to the architectures that Yann LeCun advocated in the 1990s
# for image classification

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

# This is the augmentation configuration we will use for training
# we need to make minor alterations to our existing dataset.
# Data augmentation is a way we can reduce overfitting on models, 
# where we increase the amount of training data using information only 
# in our training data

# ImageDataGenerator is a generator that can be later used for fitting a model,
# predict, etc.

train_datagen = ImageDataGenerator(
# rescales image by a factor of 1/255 
    rescale=1. / 255,
#    applying shearing transformation
    shear_range=0.1,
#   applying zoom
    zoom_range=0.2,
#   flip image
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)


# flow_from_directory reads random images in the specified subfolder
# and generates batches of augmented images 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# serialize model to JSON
modelJSON = model.to_json()
with open("5epoch20batch_model.json", "w") as jsonfile:
    jsonfile.write(modelJSON)
# serialize weights to HDF5
model.save_weights("5epoch10batch_model.h5")
with open("5epoch10batch_model.txt", "w") as textfile:
    textfile.write('5epoch10batch_model')
print("Saved model to disk")
    



#    10epoch20batch sucks
#    10epoch10batch better than ^
#    20epoch20batch results inaccurate
#    20epoch10batch best so far (1, 3, 4 right)
#    50epoch20batch shit, got 3 right
        
        
        
    
    

    
    
