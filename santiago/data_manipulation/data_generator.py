import tensorflow as tf

class DataGenerator():
    def __init__(self,path,params,shuffle=False):
        self.path=path
        self.params=params
        self.shuffle=shuffle

    #@staticmethod
    def generate(self):
        image_size = (self.params.image_height,self.params.image_width)
        #generated = tf.keras.preprocessing.image_dataset_from_directory(
        self.generated= tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        self.path,
        #labels="inferred",
        #label_mode="binary", #categorical
        #class_names=classes,
        #image_size=image_size,
        shuffle=self.shuffle,
        seed=123,
        batch_size=self.params.batch_size,

        target_size=image_size,
        class_mode='binary',
    )
        
   