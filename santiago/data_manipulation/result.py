
from os.path import join
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
import pandas as pd
import os
class Results():
    results_path = 'data/results'
    def __init__(self,result_name):
        self.result_name = result_name
        print(join(self.results_path,result_name))

    def get_results_folders(self):
        '''get the folders inside results path'''
        self.all_experiment_list = os.listdir(join(self.results_path,self.result_name))
        '''verify if the experiment is completed'''
        self.valid_experiment_list = []
        for experiment in self.all_experiment_list:
            experiment_path=join(self.results_path,self.result_name,experiment)
            if os.path.exists(join(experiment_path,'best_model','best_model.h5')):
                    if os.path.exists(join(experiment_path,'metrics_evaluation','val','val_metrics.csv')):
                        self.valid_experiment_list.append(experiment)

    def make_results_dataframe(self,checked_experiment_list,sort_by):
        val_df_list=[]
        test_df_list=[]
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        if len(checked_experiment_list)==0:
            return
        for experiment in checked_experiment_list:
            temp_test_df = pd.read_csv(join(experiment,'metrics_evaluation','test','test_metrics.csv'))
            temp_test_df['experiment_path']=experiment;temp_test_df['set']='test'
            test_df_list.append(temp_test_df)
            temp_val_df = pd.read_csv(join(experiment,'metrics_evaluation','val','val_metrics.csv'))
            temp_val_df['experiment_path']=experiment;temp_val_df['set']='val'
            val_df_list.append(temp_val_df)
        '''concat and sort'''
        self.val_df = pd.concat(val_df_list)
        self.test_df = pd.concat(test_df_list)
        self.val_df.sort_values(by=sort_by,inplace=True,ascending=False)
        self.test_df.sort_values(by=sort_by,inplace=True,ascending=False)
        

       #print(self.val_df)
    def get_widgets(self):
        self.val_widget_list = []
        self.test_widget_list = []
        for i in range(len(self.val_df)):
            self.val_widget_list.append(ResultWidget(self.val_df.iloc[i]))
        for i in range(len(self.test_df)):
            self.test_widget_list.append(ResultWidget(self.test_df.iloc[i]))

'''create class for result widget'''
class ResultWidget(QtWidgets.QWidget):
    def __init__(self,experiment):
        super(ResultWidget, self).__init__()
        loadUi('gui/result_widget.ui', self)
        self.experiment = experiment
        self.put_metrics_in_widget()
        self.display_graphs()
        
    def put_metrics_in_widget(self):
        numbers_to_show = 3
        '''put metrics in labels'''
        self.acc_label.setText("    "+str(round(self.experiment.accuracy,numbers_to_show)))
        self.precition_label.setText("    "+str(round(self.experiment.precision,numbers_to_show)))
        self.recall_label.setText("    "+str(round(self.experiment.recall,numbers_to_show)))
        self.f1_label.setText("    "+str(round(self.experiment.f1,numbers_to_show)))
        self.auc_label.setText("    "+str(round(self.experiment.auc,numbers_to_show)))
        self.FP_label.setText("    "+str(self.experiment.FP))
        self.FN_label.setText("    "+str(self.experiment.FN))
        self.TP_label.setText("    "+str(self.experiment.TP))
        self.TN_label.setText("    "+str(self.experiment.TN))
        '''change title of groupbox'''
        experiment_name=self.experiment.experiment_path.split('\\')[-1]
        self.groupBox.setTitle(experiment_name)

    def display_graphs(self):
        '''display training and validation acc and loss'''
        acc_graph_path=join(self.experiment.experiment_path,'history_results','accuracy.png')
        loss_graph_path=join(self.experiment.experiment_path,'history_results','loss.png')
        self.display_image(acc_graph_path,self.acc_graph)
        self.display_image(loss_graph_path,self.loss_graph)

        '''display testmodel graphs'''
        matrix_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'confusion_matrix.png')
        pres_rec_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'precision_recall_curve.png')
        roc_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'roc_auc_curve.png')
        self.display_image(matrix_graph_path,self.matrix_graph)
        self.display_image(pres_rec_graph_path,self.pres_rec_graph)
        self.display_image(roc_graph_path,self.roc_auc_graph)


    def display_image(self,path,label):
        self.clean_display(label)
        image_profile = QtGui.QImage(path) #QImage object
        '''get shape of self.label'''
        width = label.width()
        height = label.height()
    
        '''resize image to fit in self.label'''
        image_profile = image_profile.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  
        label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 
        
    def clean_display(self,label):
        """Clean the display"""
        label.setText('')
        '''delete pixmap from label'''
        label.setPixmap(QtGui.QPixmap(''))
    
