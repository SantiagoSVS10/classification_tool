
from santiago.data_manipulation.dataset import Dataset_initializer
from santiago.data_manipulation.result import Results
from PyQt5.QtWidgets import QApplication, QMainWindow
from santiago.model.trainer import ModelTrainer
from santiago.utils.json_params import Params
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.uic import loadUi
from os.path import join

import matplotlib.pyplot as plt
import warnings
import sys
import os

warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

"""
This is the main window of the application.
Contains visual elements that allow to train Machine Learning binary classification models. 
Allows to visualize the results, such as metrics, false positives and false negatives.

"""



class MainWindow(QMainWindow):
    datasets_path = 'data/datasets'
    results_path = 'data/results'
    gui_images_path = 'gui/images'
    params_path='data/config/params.json'
    params=Params(params_path)

    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('gui/main_window.ui', self)
        self.R_bot_frame.hide()
        self.comboOrganizer.setEnabled(False)
        self.train_status_label.setText("Not training")
        self.dataset_names=self.get_dataset_names()
        self.result_names=self.get_result_names()
        self.put_datasets_in_train_toolbox()
        self.put_results_in_analyse_toolbox()
        self.dataset_list.itemDoubleClicked.connect(self.select_dataset_from_list)
        self.result_list.itemDoubleClicked.connect(self.select_result_from_list)
        self.qt_experiments_list.itemChanged.connect(self.get_checked_experiments)
        self.save_params_button.clicked.connect(self.save_params_from_entries)
        self.train_button.clicked.connect(self.start_training_thread)
        self.distribution_button.clicked.connect(self.make_new_distribution_gui)
        self.toolBox.currentChanged.connect(self.change_between_train_and_analyse)
        
        self.comboOrganizer.currentIndexChanged.connect(self.get_checked_experiments)
        self.write_params_in_items()
        self.show()

    def select_dataset_from_list(self):
        """Select a dataset from the list"""
        self.selected_dataset_name = self.dataset_list.currentItem().text()
        
        self.current_dataset=Dataset_initializer(self.selected_dataset_name,self.params)
        plt.close('all')
        self.distributed()
    
    def select_result_from_list(self):
        """Select a result from the list"""
        self.checked_experiments=[]
        self.selected_result_name = self.result_list.currentItem().text()
        self.current_result=Results(self.selected_result_name)
        self.current_result.get_results_folders()
        self.put_experiments_in_experiments_list()
        
    def put_experiments_in_experiments_list(self):
        '''delete previous experiments'''
        self.qt_experiments_list.clear()

        '''put each experiment in the list as checkbox'''
        for experiment in self.current_result.valid_experiment_list:
            new_experiment_item=QtWidgets.QListWidgetItem(experiment)
            '''set item check with double click'''
            new_experiment_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)            
            new_experiment_item.setCheckState(QtCore.Qt.Unchecked)
            self.qt_experiments_list.addItem(new_experiment_item)

    def get_checked_experiments(self):
        """Get the checked experiments"""
        self.checked_experiments=[]
        for i in range(self.qt_experiments_list.count()):
            if self.qt_experiments_list.item(i).checkState()==2:       
                experiment_path=join(self.results_path,self.current_result.result_name,self.qt_experiments_list.item(i).text())
                self.checked_experiments.append(experiment_path)
        # print(self.checked_experiments)
        sort_by=self.comboOrganizer.currentText()
        self.current_result.make_results_dataframe(self.checked_experiments,sort_by)
        self.clean_layout(self.val_results_layout)
        self.clean_layout(self.test_results_layout)
        self.current_result.get_widgets()
        self.put_widgets()
        if len(self.checked_experiments)>0: 
            self.comboOrganizer.setEnabled(True)
            

    # def put_widgets_by_dataframe(self,experiment,layout):
    #     new_result_widget=ResultWidget(experiment)
    #     layout.addWidget(new_result_widget)
    def put_widgets(self):
        for widget in self.current_result.val_widget_list:
            self.val_results_layout.addWidget(widget)
        for widget in self.current_result.test_widget_list:
            self.test_results_layout.addWidget(widget)
    def clean_layout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
            
    def make_new_distribution_gui(self):
        """Make a new distribution of the current dataset"""
        train_percent=float(self.train_entry.text())
        val_percent=float(self.val_entry.text())
        test_percent=float(self.test_entry.text())
        self.current_dataset.create_new_distribution(train_percent,val_percent,test_percent)
        self.distributed()

    '''use the class TrainingThread to run the training in a separate thread'''
    def start_training_thread(self):
        """Start the training thread"""
        self.train_progress.setValue(0)
        self.train_status_label.setText("Training...")
        self.train_button.setEnabled(False)
        self.training_thread = TrainingThread(self.current_dataset,self.selected_dataset_name,self.params)
        self.training_thread.start()
        '''wait for the thread to finish'''
        self.training_thread.finished.connect(self.test_model)

    def test_model(self):
        """Test the model in the test and validation dataset"""
        '''use trainer object from the training thread'''
        self.train_progress.setValue(100)
        self.trainer=self.training_thread.trainer
        self.train_status_label.setText("Testing...")
        self.trainer.plot_history()
        self.trainer.test_model_with_generator(self.current_dataset,'test')
        self.trainer.test_model_with_generator(self.current_dataset,'val')

        self.result_names=self.get_result_names()
        self.put_results_in_analyse_toolbox()
        self.train_button.setEnabled(True)
        self.train_status_label.setText("Training and Test finished!")

    # def start_training(self):
    #     """Start the training"""
    #     self.current_dataset.create_data_generators()
    #     trainer = ModelTrainer(self.selected_dataset_name,self.params)
    #     trainer.train(self.current_dataset,show_plots=False)

    def write_params_in_items(self):
        self.model_combobox.setCurrentText(self.params.model)
        self.batch_entry.setText(str(self.params.batch_size))
        self.learning_rate_entry.setText(str(self.params.learning_rate))
        self.epochs_entry.setText(str(self.params.num_epochs))
        self.width_entry.setText(str(self.params.image_width))
        self.height_entry.setText(str(self.params.image_height))
        self.channels_entry.setText(str(self.params.channels))

    def save_params_from_entries(self):
        """Save the parameters from the entries"""
        self.params.model=self.model_combobox.currentText()
        self.params.learning_rate=float(self.learning_rate_entry.text())
        self.params.batch_size=int(self.batch_entry.text())
        self.params.num_epochs=int(self.epochs_entry.text())
        self.params.image_width=int(self.width_entry.text())
        self.params.image_height=int(self.height_entry.text())
        self.params.channels=int(self.channels_entry.text())
        self.params.save(self.params_path)

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

    def distributed(self):
        """Check if the current dataset is distributed"""
        self.verified_current_dataset=(self.current_dataset.verify_distribution())
        if self.verified_current_dataset==True:
            self.train_button.setEnabled(True)
            self.distribution_status.setText('Distributed')
            self.current_dataset.plot_current_training_distribution()
            self.display_image('gui/images/current_training_distribution.png',self.distribution_label)
        else:
            self.train_button.setEnabled(False)
            self.clean_display(self.distribution_label)
            self.distribution_status.setText('Not Distributed')
        return self.verified_current_dataset

    def put_datasets_in_train_toolbox(self):
        """Put the datasets in the train toolbox"""
        for dataset in self.dataset_names:
            self.dataset_list.addItem(dataset)
    
    def put_results_in_analyse_toolbox(self):
        '''clean the list'''
        self.result_list.clear()
        """Put the results in the analyse toolbox"""
        for result in self.result_names:
            self.result_list.addItem(result)

    def change_between_train_and_analyse(self):
        """Change between the train and analyse toolbox"""
        if self.toolBox.currentWidget()==self.train_toolbox:
            self.R_up_frame.show()
            self.R_bot_frame.hide()
            self.comboOrganizer.setEnabled(False)
        else:
            self.R_up_frame.hide()
            self.R_bot_frame.show()
            
    def get_dataset_names(self):
        """Get the names of the datasets in the datasets folder"""
        dataset_names = os.listdir(self.datasets_path)
        return dataset_names

    def get_result_names(self):
        """Get the names of the results in the results folder"""
        result_names = os.listdir(self.results_path)
        return result_names

# '''create class for result widget'''
# class ResultWidget(QtWidgets.QWidget):
#     def __init__(self,experiment):
#         super(ResultWidget, self).__init__()
#         loadUi('gui/result_widget.ui', self)
#         self.experiment = experiment
#         self.put_metrics_in_widget()
#         self.display_graphs()
        
#     def put_metrics_in_widget(self):
#         numbers_to_show = 3
#         '''put metrics in labels'''
#         self.acc_label.setText("    "+str(round(self.experiment.accuracy,numbers_to_show)))
#         self.precition_label.setText("    "+str(round(self.experiment.precision,numbers_to_show)))
#         self.recall_label.setText("    "+str(round(self.experiment.recall,numbers_to_show)))
#         self.f1_label.setText("    "+str(round(self.experiment.f1,numbers_to_show)))
#         self.auc_label.setText("    "+str(round(self.experiment.auc,numbers_to_show)))
#         self.FP_label.setText("    "+str(self.experiment.FP))
#         self.FN_label.setText("    "+str(self.experiment.FN))
#         self.TP_label.setText("    "+str(self.experiment.TP))
#         self.TN_label.setText("    "+str(self.experiment.TN))
#         '''change title of groupbox'''
#         experiment_name=self.experiment.experiment_path.split('\\')[-1]
#         self.groupBox.setTitle(experiment_name)

#     def display_graphs(self):
#         '''display training and validation acc and loss'''
#         acc_graph_path=join(self.experiment.experiment_path,'history_results','accuracy.png')
#         loss_graph_path=join(self.experiment.experiment_path,'history_results','loss.png')
#         self.display_image(acc_graph_path,self.acc_graph)
#         self.display_image(loss_graph_path,self.loss_graph)

#         '''display testmodel graphs'''
#         matrix_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'confusion_matrix.png')
#         pres_rec_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'precision_recall_curve.png')
#         roc_graph_path=join(self.experiment.experiment_path,'metrics_evaluation',self.experiment.set,'roc_auc_curve.png')
#         self.display_image(matrix_graph_path,self.matrix_graph)
#         self.display_image(pres_rec_graph_path,self.pres_rec_graph)
#         self.display_image(roc_graph_path,self.roc_auc_graph)


#     def display_image(self,path,label):
#         self.clean_display(label)
#         image_profile = QtGui.QImage(path) #QImage object
#         '''get shape of self.label'''
#         width = label.width()
#         height = label.height()
    
#         '''resize image to fit in self.label'''
#         image_profile = image_profile.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  
#         label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 
        
#     def clean_display(self,label):
#         """Clean the display"""
#         label.setText('')
#         '''delete pixmap from label'''
#         label.setPixmap(QtGui.QPixmap(''))

'''class to run training in a thread'''
class TrainingThread(QThread):
    def __init__(self,current_dataset,selected_dataset_name,params):
        QThread.__init__(self)
        self.current_dataset=current_dataset
        self.selected_dataset_name=selected_dataset_name
        self.params=params
        
    def run(self):
        self.start_training()

    def start_training(self):
        """Start the training"""
        self.current_dataset.create_data_generators()
        self.trainer = ModelTrainer(self.selected_dataset_name,self.params)
        self.trainer.train(self.current_dataset,window,show_plots=False,save_plots=True)
        
        self.exit()

        

    def test_model(self):
        """Test the model in the test and validation dataset"""
        self.trainer.plot_history()
        self.trainer.test_model_with_generator(self.current_dataset,'test')
        self.trainer.test_model_with_generator(self.current_dataset,'val')
        print('Finished!!')
        #window.train_progress.setValue(100)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())