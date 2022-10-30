from santiago.model.models import ModelManager
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
from santiago.utils.train_utils import PlotLearning,SavePlotLearning
from os.path import join,exists
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()

import pandas as pd
import numpy as np
import tensorflow as tf
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

        self.metrics_evaluation_path = join(self.experiment_folder, "metrics_evaluation")
        self.test_metrics_path = join(self.metrics_evaluation_path, "test")
        self.val_metrics_path = join(self.metrics_evaluation_path, "val")
        self.history_training_path = join(self.experiment_folder, "history_results")
        self.best_model_path = join(self.experiment_folder, "best_model")
        '''make dirs'''
        if not exists(self.metrics_evaluation_path):
            os.mkdir(self.metrics_evaluation_path)
            os.mkdir(self.test_metrics_path)
            os.mkdir(self.val_metrics_path)
        if not exists(self.history_training_path):
            os.mkdir(self.history_training_path)
        if not exists(self.best_model_path):
            os.mkdir(self.best_model_path)
        

        # Define filepaths with names
        self.h5_best_model_filepath = join(
            self.best_model_path, 'best_model.h5')
        self.history_training_csv_filepath = join(
            self.history_training_path, 'training_history_log.csv')


    def train(self, dataset,window=None,show_plots=False,save_plots=False):
        self.create_callbacks(window,show_plots,save_plots)
        self.compile_model()
        self.model.summary()
        self.history=self.model.fit(dataset.train_generator,
                    steps_per_epoch=len(dataset.train_generator),
                    epochs=self.params.num_epochs,
                    validation_data=dataset.val_generator,
                    validation_steps=len(dataset.val_generator),
                    callbacks=self.callbacks_list)
                       
    def test_model_with_generator(self,dataset,set_name):
        
        '''get y_true and y_pred'''
        if set_name=='test':
            save_path=self.test_metrics_path
            generator=dataset.test_generator
            filepaths=dataset.test_generator.filepaths
        elif set_name=='val':
            save_path=self.val_metrics_path
            generator=dataset.val_generator
            filepaths=dataset.val_generator.filepaths

        '''get y_true from generator'''
        y_true = generator.classes
        y_pred = self.model.predict(generator,steps=len(generator),verbose=1)
        for i in range(len(y_pred)):
            if y_pred[i]>=0.5:
                y_pred[i]=1
            else:
                y_pred[i]=0

        '''convert y_pred to one dimensional'''
        y_pred = y_pred.flatten()

        '''get metrics'''
        accuracy=accuracy_score(y_true,y_pred)
        presition=precision_score(y_true,y_pred,average="macro")
        recall=recall_score(y_true,y_pred,average="macro")
        f1=f1_score(y_true,y_pred,average="macro",zero_division=0)
        auc=roc_auc_score(y_true,y_pred)
        average_precision=average_precision_score(y_true,y_pred)
        matrix_confusion = (confusion_matrix(y_true, y_pred))
        report = classification_report(y_true, y_pred, target_names=dataset.classes,digits=3,output_dict=True)

        self.precision_recall_curves(save_path,set_name,y_true,y_pred)
        #auc_roc(results_dir,setname,true_categories_tf, test_predictions_tf)

        self.make_csv(save_path,filepaths,y_true,y_pred)

        plt.close()
        plt.figure(figsize=(10, 10))
        
        sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('predicted value')
        plt.ylabel('true value')
        plt.savefig(join(save_path,"confusion_matrix.png"),bbox_inches='tight')

        plt.close()
        #cr=classification_report(true_categories, test_predictions, target_names=params.classes, digits=3,output_dict=True)
        df_cr=pd.DataFrame(report).transpose()
        df_cr.to_csv(join(save_path,"classification_report.csv"))

    def make_csv(self,save_path,filenames,y_true,y_pred):
        batch_csv=pd.DataFrame()
        batch_csv['filename'] = pd.Series(filenames)
        batch_csv['y_true'] = pd.Series(y_true)
        batch_csv['y_pred'] = pd.Series(y_pred)
        batch_csv.to_csv(join(save_path,'classificated_images.csv'),index=False)

    def precision_recall_curves(self,save_path,set_name,true_categories_tf,test_predictions_tf):
        plt.close()
        plt.figure(figsize=(10, 10))
        precision, recall, _ = precision_recall_curve(true_categories_tf, test_predictions_tf)
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig(join(save_path,"precision_recall_curve.png"),bbox_inches='tight')

    def plot_history(self):

        plt.close()
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        acc_fig=plt.savefig(join(self.history_training_path,"accuracy"+".png"),bbox_inches='tight')
        plt.close()

        '''plot in main thread'''
        
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig(results_dir+"/loss_acc.png")
        loss_fig=plt.savefig(join(self.history_training_path,"loss"+".png"),bbox_inches='tight')
        plt.close()
        return acc_fig,loss_fig

    def create_callbacks(self,window=None, show_learning_curves=False,save_plots=False):
        
        early_stop = EarlyStopping(monitor='val_loss', patience=100)
        
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
        if(save_plots):
            save_losses = SavePlotLearning(window)
            self.callbacks_list.append(save_losses)