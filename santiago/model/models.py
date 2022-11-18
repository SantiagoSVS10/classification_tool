import traceback
import tensorflow as tf
import os

from classification_models.tfkeras import Classifiers
import efficientnet.tfkeras as efn


class ModelManager():
    """
    Class manager for diferent classification models 
    """
    __classification_models__ = Classifiers.models_names(
    )  # models in classification_models library
    __effn_models__ = [('EfficientNetB' + str(i))
                       for i in range(8)]  # EfficientNetB0 --> EfficientNetB7
    __default_network__ = 'EfficientNetB0'  # should be in the efn library

    def __init__(self,params):
        self.hparams = params
        
        
    #@staticmethod
    def load_model(hparams, path):
        model = ModelManager.create_model(hparams)
        model.load_weights(path)
        return model

    #@staticmethod
    def create_model(self):
        '''method to call the appropiate model creation according to hparams configuration file'''
        model_type = None

        # Available models and libraries
        switch = {
            'effn_model': ModelManager.__create_effn_model__,  # from EfficientNet library
            # from classification models library
            'cls_model': ModelManager.__create_cls_model__,
            'custom_model': ModelManager.__create_custom_model__,  # Keras sequential model
            # Here can be inserted other functions to create keras models
        }

        # Check if suppported by Effn library
        if(self.hparams.model.lower() in map(str.lower, ModelManager.__effn_models__)):
            model_type = 'effn_model'

        # Check if suppported classification models library
        if(self.hparams.model.lower() in map(str.lower, ModelManager.__classification_models__)):
            model_type = 'cls_model'

        if(self.hparams.model == 'custom_model'):
            model_type = 'custom_model'

        model = None
        try:
            if(model_type in switch):
                model = switch[model_type](self.hparams)
        except Exception as e:
            traceback.print_exc()
            model = None

        if(model is None):
            self.hparams.model = ModelManager.__default_network__
            model = switch['effn_model'](self.hparams)

        self.model=model

    # ---------------------------------------------------------------------------- #
    #                         functions for model creation                         #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def __create_effn_model__(hparams, n_channels=3):
        # model_name = hparams.model
        n_channels = hparams.channels
        index_model = next(i for i, v in enumerate(
            ModelManager.__effn_models__) if v.lower() == hparams.model.lower())
        model_name = ModelManager.__effn_models__[index_model]
        # Gets a callable function from model name on hparams
        model_function = getattr(efn, model_name)

        # General form of calling "tf.keras.applications.EfficientNetB0", working for each available EfficientNet model
        conv_base = model_function(input_shape=(hparams.image_height, hparams.image_width, n_channels),
                                   include_top=False,
                                   weights=None)

        x = conv_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

        output_layer = tf.keras.layers.Dense(
            hparams.n_classes, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

        return model

    @staticmethod
    def __create_cls_model__(hparams, n_channels=1):
        n_channels = hparams.channels
        print(f'creating {hparams.model} model')
        # model_name = hparams.model
        index_model = next(i for i, v in enumerate(
            ModelManager.__classification_models__) if v.lower() == hparams.model.lower())
        model_name = ModelManager.__classification_models__[index_model]
        activ_func = 'sigmoid'
        model_instance, _ = Classifiers.get(model_name)

        conv_base = model_instance(input_shape=(hparams.image_width, hparams.image_height, n_channels),
                                   weights=None,
                                   include_top=False
                                   )

        x = conv_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        output_layer = tf.keras.layers.Dense(
            hparams.n_classes, activation=activ_func)(x)
        model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

        return model

    @staticmethod
    def __create_custom_model__(hparams, n_channels=1):
        ActivationRelu = tf.keras.activations.relu

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=ActivationRelu,
                                         input_shape=(hparams.image_width, hparams.image_height, 3)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(
            32, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(
            64, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(
            64, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(
            128, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(activation=ActivationRelu, units=512))
        model.add(tf.keras.layers.Dense(activation=ActivationRelu, units=64))
        model.add(tf.keras.layers.Dense(activation=ActivationRelu, units=16))
        model.add(tf.keras.layers.Dense(
            activation="sigmoid", units=hparams.n_classes))

        return model