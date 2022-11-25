
from os.path import join
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
import pandas as pd
import os
class Results():
    results_path = 'data/results'
    def __init__(self,result_name):
        self.result_name = result_name

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
        self.val_images_df_list=[]
        self.test_images_df_list=[]
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        if len(checked_experiment_list)==0:
            return
        for experiment in checked_experiment_list:
            '''get test and validation dataframes'''
            temp_test_df = pd.read_csv(join(experiment,'metrics_evaluation','test','test_metrics.csv'))
            temp_test_df['experiment_path']=experiment;temp_test_df['set']='test'
            test_df_list.append(temp_test_df)
            temp_val_df = pd.read_csv(join(experiment,'metrics_evaluation','val','val_metrics.csv'))
            temp_val_df['experiment_path']=experiment;temp_val_df['set']='val'
            val_df_list.append(temp_val_df)

            '''get images dataframe'''
            val_temp_images_df = pd.read_csv(join(experiment,'metrics_evaluation','val','classificated_images.csv'))
            val_temp_images_df['experiment_path']=experiment;val_temp_images_df['set']='val'
            self.val_images_df_list.append(val_temp_images_df)
            test_temp_images_df = pd.read_csv(join(experiment,'metrics_evaluation','test','classificated_images.csv'))
            test_temp_images_df['experiment_path']=experiment;test_temp_images_df['set']='test'
            self.test_images_df_list.append(test_temp_images_df)
            
        '''concat and sort'''
        self.val_df = pd.concat(val_df_list)
        self.test_df = pd.concat(test_df_list)
        self.val_df.sort_values(by=sort_by,inplace=True,ascending=False)
        self.test_df.sort_values(by=sort_by,inplace=True,ascending=False)
        
        # self.val_images=pd.concat(val_images_df_list)
        # self.test_images=pd.concat(test_images_df_list)
        
   
    def get_widgets(self):
        self.val_widget_list = []
        self.test_widget_list = []
        for i in range(len(self.val_df)):
            self.val_widget_list.append(ResultWidget(self.val_df.iloc[i]))
        for i in range(len(self.test_df)):
            self.test_widget_list.append(ResultWidget(self.test_df.iloc[i]))

    def get_images_widgets(self,filter,set):
        if set=='Validation':
            df_list = self.val_images_df_list
        elif set=='Test':
            df_list = self.test_images_df_list
        self.image_widgets=[]
        for image_df in df_list:
            image_df=image_df[image_df['classification']==filter]
            if len(image_df)<1:
                return
            image_df=image_df[0:10]
            images_widget=QtWidgets.QWidget()
            v_lay=QtWidgets.QVBoxLayout()
            '''put elements in top of the layout'''
            v_lay.setAlignment(QtCore.Qt.AlignTop)
            title=QtWidgets.QLabel(image_df.iloc[0].experiment_path.split('\\')[-1])
            v_lay.addWidget(title)
            for i in range(len(image_df)):
                image_label=QtWidgets.QLabel()
                v_lay.addWidget(image_label)                
                self.display_image(image_df.iloc[i].filename,image_label)
                v_lay.addSpacerItem(QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding))
                images_widget.setLayout(v_lay)
                '''add spacer to widget'''
                

            self.image_widgets.append(images_widget)

    def display_image(self,path,label):
        self.clean_display(label)
        image_profile = QtGui.QImage(path) #QImage object
        '''get shape of self.label'''
        '''set width of label'''
        label.setFixedWidth(400)

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
    
