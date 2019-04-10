from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import glob


class ClassifyImage():
    
    def loadModelandWeights(self, modelName):
        # load json and create model
        JSONfile = open(modelName+".json", "r")
        loadedModelJSON = JSONfile.read()
        JSONfile.close()
        loadedModel = model_from_json(loadedModelJSON)
        # load weights into new model
        loadedModel.load_weights(modelName+".h5")
        loadedModel.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        print("Loaded model from disk")
        return loadedModel
        
    def classifyImage(self, model, img):
    #        newImg = image.load_img(img, target_size = (250,250))
        newImg = image.img_to_array(img)
        newImg = np.expand_dims(newImg, axis=0)
        result = model.predict(newImg)
        
#        test = self.model.predict(img)
#            print(result)
    #        classification = self.train_generator.class_indices
    #        print(classification)
        
        if result[0][0] == 1.:
            return('male')
        else: return('female')
        
    def testingAccuracy(self, modelName, dataPath, gender='male'):
        filenames = glob.glob(dataPath)
        filenames.sort()
        images = [image.load_img(img, target_size = (250,250)) for img in filenames]
        
#        ci = ClassifyImage.ClassifyImage()
        model = self.loadModelandWeights(modelName)
        results = []
        for i in range(len(images)):
            results.append(self.classifyImage(model, images[i]))
            
        
        print('{} out of {}'.format(results.count('male'), len(filenames)))

        