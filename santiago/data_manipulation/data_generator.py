import tensorflow as tf

class DataGenerator(object):
    def __init__(self):
        pass

    @staticmethod
    def generate(path,classes,params,shuffle=False):
        image_size = (params.image_height,params.image_width)
        #generated = tf.keras.preprocessing.image_dataset_from_directory(
        generated= tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        path,
        #labels="inferred",
        #label_mode="binary", #categorical
        #class_names=classes,
        #image_size=image_size,
        shuffle=shuffle,
        seed=123,
        batch_size=params.batch_size,
        target_size=image_size,
        class_mode='binary',
    )
        return generated
   