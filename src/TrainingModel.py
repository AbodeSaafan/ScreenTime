from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K



# dimensions of our images
img_width, img_height = 250, 250

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 364
nb_validation_samples = 52
epochs = 10
batch_size = 20
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    

model = Sequential()

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

# this is the augmentation configuration we will use for training
#  we need to make minor alterations to our existing dataset
# Data augmentation is a way we can reduce overfitting on models, where we increase the amount of training data using information only in our training data

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
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

#model.save_weights('50epoch10batch.h5')

# serialize model to JSON
modelJSON = model.to_json()
with open("10epoch20batch_model.json", "w") as jsonfile:
    jsonfile.write(modelJSON)
# serialize weights to HDF5
model.save_weights("10epoch20batch_model.h5")
with open("10epoch20batch_model.txt", "w") as textfile:
    textfile.write('10epoch20batch_model')
print("Saved model to disk")
    






#def calculateTestAccuracy(self, img):
##        img = image.load_img(i, target_size = (250,250))
#    testImg = image.img_to_array(img)
#    testImg = np.expand_dims(testImg, axis=0)
#    result = model.predict(testImg)
#    print(result)
    
#    10epoch20batch sucks
#    10epoch10batch better than ^
#    20epoch20batch results inaccurate
#    20epoch10batch best so far (1, 3, 4 right)
#    50epoch20batch shit 3 right
#    50epoch10batch
        
        
        
    
    
    
    
        
#    def classifyImage(self,img):
#       
#        self.model.load_weights('weights.h5')
##        newImg = image.load_img(img, target_size = (250,250))
#        newImg = image.img_to_array(img)
#        newImg = np.expand_dims(newImg, axis=0)
#        result = self.model.predict(newImg)
##        classification = self.train_generator.class_indices
##        print(classification)
#        if result[0][0] == 1.:
#            print('male')
#        else: print('female')
        
        
        
        
#        for i in test_images:
#            pic = plt.imread(i, 0)
#            plt.imshow(pic)
#            plt.show()
#            img = image.load_img(i, target_size = (250,250))
#            img = image.img_to_array(img)
#            img = np.expand_dims(img, axis=0)
#            result = model.predict(img)
#            print(result)
#            temp = train_generator.class_indices
#            if result[0][0] == 1.:
#                print('female')
#            else: print('male')
        
    
    
    
