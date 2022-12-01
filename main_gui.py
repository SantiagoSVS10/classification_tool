
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
        self.set_groupbox.setEnabled(False)
        self.comboClassification.setVisible(False)
        self.train_status_label.setText("Not training")
        self.dataset_names=self.get_dataset_names()
        self.result_names=self.get_result_names()
        '''put function to tabWidget'''
        self.tabWidget.currentChanged.connect(self.tab_changed)
        self.put_datasets_in_train_toolbox()
        self.put_results_in_analyse_toolbox()
        self.checked_set='Validation'
        self.set_groupbox.children()[1].clicked.connect(self.get_checked_set)    
        self.set_groupbox.children()[2].clicked.connect(self.get_checked_set)
        self.dataset_list.itemDoubleClicked.connect(self.select_dataset_from_list)
        self.result_list.itemDoubleClicked.connect(self.select_result_from_list)
        self.qt_experiments_list.itemChanged.connect(self.get_checked_experiments)
        self.save_params_button.clicked.connect(self.save_params_from_entries)
        self.train_button.clicked.connect(self.start_training_thread)
        self.distribution_button.clicked.connect(self.make_new_distribution_gui)
        self.toolBox.currentChanged.connect(self.change_between_train_and_analyse)
        
        self.comboOrganizer.currentIndexChanged.connect(self.get_checked_experiments)
        self.comboClassification.currentIndexChanged.connect(self.get_checked_experiments)
        self.write_params_in_items()
        self.show()

    '''function to select dataset from list'''
    def select_dataset_from_list(self):
        """Select a dataset from the list"""
        self.selected_dataset_name = self.dataset_list.currentItem().text()
        
        self.current_dataset=Dataset_initializer(self.selected_dataset_name,self.params)
        plt.close('all')
        self.current_dataset.create_distribution_dataframe()
        self.distributed()
    
    '''function to select experiments from list'''
    def select_result_from_list(self):
        """Select a result from the list"""
        self.checked_experiments=[]
        self.selected_result_name = self.result_list.currentItem().text()
        self.current_result=Results(self.selected_result_name)
        self.current_result.get_results_folders()
        self.put_experiments_in_experiments_list()
    
    '''function to read experiment folders and put them in the list'''
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

    '''function to identify a tab change'''
    def tab_changed(self):
        '''clean layout when changing tab'''
        if self.tabWidget.currentIndex()==1:
            self.label.setText("Filter:")
            self.comboClassification.setVisible(True)
            self.comboOrganizer.setVisible(False)
        elif self.tabWidget.currentIndex()==0:
            self.label.setText("Organize experiments by:")
            self.comboClassification.setVisible(False)
            self.comboOrganizer.setVisible(True)

    '''function to get checked set (validation or test)'''
    def get_checked_set(self):
        '''get checked item in groupbox by children'''
        for i in range(self.set_groupbox.layout().count()):
            if self.set_groupbox.layout().itemAt(i).widget().isChecked():
                self.checked_set=self.set_groupbox.layout().itemAt(i).widget().text()
        self.get_checked_experiments()
        
    '''function to get the checked experiments in the gui'''
    def get_checked_experiments(self):
        """Get the checked experiments"""
        self.checked_experiments=[]
        for i in range(self.qt_experiments_list.count()):
            if self.qt_experiments_list.item(i).checkState()==2:       
                experiment_path=join(self.results_path,self.current_result.result_name,self.qt_experiments_list.item(i).text())
                self.checked_experiments.append(experiment_path)

        sort_by=self.comboOrganizer.currentText()
        filter=self.comboClassification.currentText()
        self.clean_layout(self.val_results_layout)
        self.clean_layout(self.h_lay_images) 
        self.current_result.make_results_dataframe(self.checked_experiments,sort_by)
        
        self.current_result.get_widgets()
        self.put_widgets()
    
        self.current_result.get_images_widgets(filter,self.checked_set)
        self.put_individual_images()

        if len(self.checked_experiments)>0: 
            self.comboOrganizer.setEnabled(True)
            self.set_groupbox.setEnabled(True)

    '''function to show the experiment widgets'''
    def put_widgets(self):
        if self.checked_set=='Validation':
            for widget in self.current_result.val_widget_list:
                self.val_results_layout.addWidget(widget)
        elif self.checked_set=='Test':
            for widget in self.current_result.test_widget_list:
                self.val_results_layout.addWidget(widget)
    
    '''function to put the individual images in a layout'''
    def put_individual_images(self):
        '''create a vertical layout for each checked experiment'''
        for widget in self.current_result.image_widgets:
            '''creake vertical layout'''
            self.h_lay_images.addWidget(widget)

    '''function to clean a layout'''
    def clean_layout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
    '''function to get the values from the entries and create new distribution'''
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

    '''function to start model testing'''
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

    '''function to read params and put them in the gui'''
    def write_params_in_items(self):
        self.model_combobox.setCurrentText(self.params.model)
        self.batch_entry.setText(str(self.params.batch_size))
        self.learning_rate_entry.setText(str(self.params.learning_rate))
        self.epochs_entry.setText(str(self.params.num_epochs))
        self.width_entry.setText(str(self.params.image_width))
        self.height_entry.setText(str(self.params.image_height))
        self.channels_entry.setText(str(self.params.channels))

    '''function to update the params using the values in the gui'''
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

    '''function to display an image in a label'''
    def display_image(self,path,label):
        self.clean_display(label)
        image_profile = QtGui.QImage(path) #QImage object
        '''get shape of self.label'''
        width = label.width()
        height = label.height()
        
        '''resize image to fit in self.label'''
        image_profile = image_profile.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)  
        label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 

    '''function to clean one of the labels'''
    def clean_display(self,label):
        """Clean the display"""
        label.setText('')
        '''delete pixmap from label'''
        label.setPixmap(QtGui.QPixmap(''))

    '''function to check the type of distribution'''
    def distributed(self):
        """Check if the current dataset is distributed"""
        
        if self.current_dataset.distributed=='folders':
            self.train_button.setEnabled(True)
            self.distribution_status.setText('Distributed')
            self.current_dataset.plot_current_training_distribution()
            self.display_image('gui/images/current_training_distribution.png',self.distribution_label)
        elif self.current_dataset.distributed=='dataframe':
            
            self.current_dataset.plot_current_training_distribution_dataframe()
            self.display_image('gui/images/current_training_distribution.png',self.distribution_label)
            self.train_button.setEnabled(True)
            self.distribution_status.setText('Distributed')
        else:
            self.train_button.setEnabled(False)
            self.clean_display(self.distribution_label)
            self.distribution_status.setText('Not Distributed')
        

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

'''Initialize the app'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())