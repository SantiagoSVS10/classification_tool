from santiago.model.models import ModelManager
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
from santiago.utils.train_utils import PlotLearning,SavePlotLearning
from os.path import join,exists
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, accuracy_score,PrecisionRecallDisplay

import seaborn as sns
import matplotlib.pyplot as plt
plt.ioff()

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
class ModelTrainer():
    """
    Manage the models, compiles, show summary and train the model
    """
    def __init__(self,dataset_name, params):
        self.params = params
        self.dataset_name = dataset_name
        self.create_model_architecture()
        self.experiments_result_folder=join('data/results',self.dataset_name)
        self.define_namespace()
        self.callbacks_list = []

    def create_model_architecture(self):
        self.model_manager = ModelManager(self.params)
        self.model_manager.create_model()
        self.model=self.model_manager.model
        

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
        self.final_history_training_csv_filepath = join(
            self.history_training_path, 'final_training_history_log.csv')


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
                       

        '''save history'''
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(self.final_history_training_csv_filepath, index=False)

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
        y_pred = y_pred.flatten()
        self.precision_recall_curves(save_path,set_name,y_true,y_pred)
        self.roc_auc_curves(save_path,set_name,y_true,y_pred)

        for i in range(len(y_pred)):
            if y_pred[i]>=0.5:
                y_pred[i]=1
            else:
                y_pred[i]=0
        
        '''get metrics'''
        accuracy=accuracy_score(y_true,y_pred)
        precision=precision_score(y_true,y_pred,zero_division=1)
        recall=recall_score(y_true,y_pred,zero_division=1)
        f1=f1_score(y_true,y_pred,zero_division=1)
        auc=roc_auc_score(y_true,y_pred)
        matrix_confusion = (confusion_matrix(y_true, y_pred))
        FP=matrix_confusion[0][1]
        FN=matrix_confusion[1][0]
        TP=matrix_confusion[1][1]
        TN=matrix_confusion[0][0]

        '''make csv with metrics'''
        metrics_df = pd.DataFrame({'accuracy':accuracy,
                                      'precision':precision,
                                      'recall':recall,
                                      'f1':f1,
                                      'auc':auc,
                                      'FP':FP,
                                      'FN':FN,
                                      'TP':TP,
                                      'TN':TN},index=[0])
        metrics_df.to_csv(join(save_path,set_name+'_metrics.csv'), index=False)

        '''print metrics'''
        print('Accuracy: ',accuracy)
        print('Precision: ',precision)
        print('Recall: ',recall)
        print('F1: ',f1)
        print('AUC: ',auc)
        print('False Positive: ',FP)
        print('False Negative: ',FN)
        print('True Positive: ',TP)
        print('True Negative: ',TN)

        '''make list of FP,FN,TP,TN with y_true and y_pred'''
        classification=[]
        for i in range(len(y_true)):
            if y_true[i]==0 and y_pred[i]==0:
                classification.append('TN')
            elif y_true[i]==0 and y_pred[i]==1:
                classification.append('FP')
            elif y_true[i]==1 and y_pred[i]==0:
                classification.append('FN')
            elif y_true[i]==1 and y_pred[i]==1:
                classification.append('TP')
        '''make csv with y_true,y_pred and classification'''
        self.make_csv(save_path,filepaths,y_true,y_pred,classification)

        plt.close()
        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.xlabel('predicted value')
        plt.ylabel('true value')
        plt.savefig(join(save_path,"confusion_matrix.png"),bbox_inches='tight')
        plt.close()
        #cr=classification_report(true_categories, test_predictions, target_names=params.classes, digits=3,output_dict=True)
        
        
    def make_metrics_csv(self,save_path,metrics):
        pass
    def make_csv(self,save_path,filenames,y_true,y_pred,classification):
        batch_csv=pd.DataFrame()
        batch_csv['filename'] = pd.Series(filenames)
        batch_csv['y_true'] = pd.Series(y_true)
        batch_csv['y_pred'] = pd.Series(y_pred)
        batch_csv['classification'] = pd.Series(classification)

        batch_csv.to_csv(join(save_path,'classificated_images.csv'),index=False)

    '''function to calculate TP,FP,TN,FN'''
    def calculate_TP_FP_TN_FN(self,y_true,y_pred):
        TP=0
        FP=0
        TN=0
        FN=0
        for i in range(len(y_true)):
            if y_true[i]==1 and y_pred[i]==1:
                TP+=1
            elif y_true[i]==0 and y_pred[i]==1:
                FP+=1
            elif y_true[i]==0 and y_pred[i]==0:
                TN+=1
            elif y_true[i]==1 and y_pred[i]==0:
                FN+=1
        return TP,FP,TN,FN

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

        '''plot another precision recall curve'''
        plt.close()
        plt.figure(figsize=(10, 10))
        display = PrecisionRecallDisplay.from_predictions(true_categories_tf, test_predictions_tf, name="LinearSVC")
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        plt.savefig(join(save_path,"precision_recall_curve_2.png"),bbox_inches='tight')
        plt.close()

    def roc_auc_curves(self,save_path,set_name,true_categories_tf,test_predictions_tf):
        plt.close()
        plt.figure(figsize=(10, 10))
        fpr, tpr, _ = roc_curve(true_categories_tf, test_predictions_tf)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(join(save_path,"roc_auc_curve.png"),bbox_inches='tight')

    def plot_history(self):

        plt.close()
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        acc_fig=plt.savefig(join(self.history_training_path,"accuracy"+".png"),bbox_inches='tight')
        plt.close()

        '''plot in main thread'''
        
        plt.figure(figsize=(10, 10))
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.savefig(results_dir+"/loss_acc.png")
        loss_fig=plt.savefig(join(self.history_training_path,"loss"+".png"),bbox_inches='tight')
        plt.close()
        return acc_fig,loss_fig

    def create_callbacks(self,window=None, show_learning_curves=False,save_plots=False):
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        
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