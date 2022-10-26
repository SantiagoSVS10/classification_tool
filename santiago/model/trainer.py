from santiago.model.models import ModelManager
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger

from os.path import join,exists
import time
import os
class ModelTrainer(object):
    """
    Manage the models, compiles, show summary and train the model
    """
    def __init__(self,dataset_name, params):
        self.params = params
        self.dataset_name = dataset_name
        self.model = self.create_model_architecture()
        self.experiments_result_folder=join('data/results',self.dataset_name)
        self.define_namespace()
        self.callbacks_list = []

    def create_model_architecture(self):
        model = ModelManager.create_model(self.params)
        return model

    def compile_model(self):
        self.model.compile(loss=self.params.loss,
                           optimizer=self.params.optimizer,
                           metrics=['accuracy'])
    def define_namespace(self):
        '''create metrics folder inside self.experiment_result_folder'''
        self.experiment_folder=join(self.experiments_result_folder,self.dataset_name+"_"+self.params.model + time.strftime("_%Y_%m%d_%H%M"))
        if not exists(self.experiment_folder):
            os.mkdir(self.experiment_folder)

        self.metrics_evaluation_path = join(
            self.experiment_folder, "metrics_evaluation")
        self.history_training_path = join(
            self.experiment_folder, "history_results")
        self.best_model_path = join(self.experiment_folder, "best_model")
        '''make dirs'''
        if not exists(self.metrics_evaluation_path):
            os.mkdir(self.metrics_evaluation_path)
        if not exists(self.history_training_path):
            os.mkdir(self.history_training_path)
        if not exists(self.best_model_path):
            os.mkdir(self.best_model_path)
        

        # Define filepaths with names
        self.h5_best_model_filepath = join(
            self.best_model_path, 'best_model.h5')
        self.history_training_csv_filepath = join(
            self.history_training_path, 'training_history_log.csv')


    def train(self, dataset,show_plots=False):
        self.create_callbacks(show_plots)
        self.compile_model()
        self.model.summary()
        self.model.fit(dataset.train_generator,
                    steps_per_epoch=len(dataset.train_generator),
                    epochs=self.params.num_epochs,
                    validation_data=dataset.val_generator,
                    validation_steps=len(dataset.val_generator),
                    callbacks=self.callbacks_list)
                       

    def create_callbacks(self, show_learning_curves=False):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        if(self.params.save_model_results):
            model_checkpoint = ModelCheckpoint(
                self.h5_best_model_filepath, save_weights_only=True, save_best_only=True, mode='auto')
            csv_logger = CSVLogger(self.history_training_csv_filepath)

            self.callbacks_list = [early_stop, model_checkpoint, csv_logger]
        else:
            self.callbacks_list = [early_stop]

        if(show_learning_curves):
            plot_losses = PlotLearning()
            self.callbacks_list.append(plot_losses)
