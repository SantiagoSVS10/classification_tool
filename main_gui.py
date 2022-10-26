from PyQt5.QtCore import QObject, QThread, pyqtSignal
from santiago.data_manipulation.dataset import Dataset_initializer
from santiago.utils.json_params import Params
from santiago.model.trainer import ModelTrainer
from PyQt5.QtWidgets import QApplication, QMainWindow 

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
import sys
import os

"""
This is the main window of the application.
Contains visual elements to train and analyse Machine Learning classification models.

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
        self.dataset_names=self.get_dataset_names()
        self.result_names=self.get_result_names()
        self.put_datasets_in_train_toolbox()
        self.put_results_in_analyse_toolbox()
        self.dataset_list.itemDoubleClicked.connect(self.select_dataset_from_list)
        self.save_params_button.clicked.connect(self.save_params_from_entries)
        self.write_params_in_items()
        self.label.setText(str(self.result_names))
        self.train_button.clicked.connect(self.start_training_thread)
        self.show()

    def select_dataset_from_list(self):
        """Select a dataset from the list"""
        self.selected_dataset_name = self.dataset_list.currentItem().text()
        
        self.current_dataset=Dataset_initializer(self.selected_dataset_name,self.params)
        
        self.distributed()
        
    '''use the class TrainingThread to run the training in a separate thread'''
    def start_training_thread(self):
        """Start the training thread"""
        self.training_thread = TrainingThread(self.current_dataset,self.selected_dataset_name,self.params)
        self.training_thread.start()

    def start_training(self):
        """Start the training"""
        self.current_dataset.create_data_generators()
        trainer = ModelTrainer(self.selected_dataset_name,self.params)
        trainer.train(self.current_dataset,show_plots=False)


    def write_params_in_items(self):
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

    def save_new_params(self):
        """ Save the new parameters in the params file"""
        print(self.params.learning_rate)
        print(self.params.batch_size)
        self.params.learning_rate=0.0002
        print(self.params.learning_rate)
        print(self.params.batch_size)
        self.params.save(self.params_path)

    def display_image(self):
        image_profile = QtGui.QImage('gui/images/current_training_distribution.png') #QImage object
        '''get shape of self.label'''
        width = self.distribution_label.width()
        height = self.distribution_label.height()
        '''resize image to fit in self.label'''
        image_profile = image_profile.scaled(width, height, QtCore.Qt.KeepAspectRatio)  
        self.distribution_label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 
    
    def clean_display(self):
        """Clean the display"""
        self.distribution_label.setText('')
        '''delete pixmap from label'''
        self.distribution_label.setPixmap(QtGui.QPixmap(''))

    def distributed(self):
        """Check if the current dataset is distributed"""
        self.verified_current_dataset=(self.current_dataset.verify_distribution())
        if self.verified_current_dataset==True:
            self.label.setText('Distributed')
            self.current_dataset.plot_current_training_distribution()
            self.display_image()
        else:
            self.clean_display()
            self.label.setText('Not Distributed')
        return self.verified_current_dataset

    def put_datasets_in_train_toolbox(self):
        """Put the datasets in the train toolbox"""
        for dataset in self.dataset_names:
            self.dataset_list.addItem(dataset)
    
    def put_results_in_analyse_toolbox(self):
        """Put the results in the analyse toolbox"""
        for result in self.result_names:
            self.result_list.addItem(result)

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
        trainer = ModelTrainer(self.selected_dataset_name,self.params)
        trainer.train(self.current_dataset,show_plots=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
    