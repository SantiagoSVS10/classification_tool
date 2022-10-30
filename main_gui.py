from PyQt5.QtCore import QObject, QThread, pyqtSignal
from santiago.data_manipulation.dataset import Dataset_initializer
from santiago.data_manipulation.result import Results
from santiago.utils.json_params import Params
from santiago.model.trainer import ModelTrainer
from PyQt5.QtWidgets import QApplication, QMainWindow 


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.ioff()

from PyQt5.uic import loadUi
from watchdog.observers import Observer
import warnings
import time
import sys
import os

warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

"""
This is the main window of the application.
Contains visual elements to train and analyse Machine Learning classification models.

"""
'''delete white borders from matplotlib plots'''
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
'''delete external borders from matplotlib plots'''
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'


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
        self.dataset_names=self.get_dataset_names()
        self.result_names=self.get_result_names()
        self.put_datasets_in_train_toolbox()
        self.put_results_in_analyse_toolbox()
        self.dataset_list.itemDoubleClicked.connect(self.select_dataset_from_list)
        self.result_list.itemDoubleClicked.connect(self.select_result_from_list)
        self.save_params_button.clicked.connect(self.save_params_from_entries)
        self.train_button.clicked.connect(self.start_training_thread)
        self.distribution_button.clicked.connect(self.make_new_distribution_gui)
        self.toolBox.currentChanged.connect(self.change_between_train_and_analyse)
        #self.star_thread_code()
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
        self.selected_result_name = self.result_list.currentItem().text()
        self.current_result=Results(self.selected_result_name)
        self.current_result.get_results_folders()

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
        self.training_thread = TrainingThread(self.current_dataset,self.selected_dataset_name,self.params)
        self.training_thread.start()

    # def start_training(self):
    #     """Start the training"""
    #     self.current_dataset.create_data_generators()
    #     trainer = ModelTrainer(self.selected_dataset_name,self.params)
    #     trainer.train(self.current_dataset,show_plots=False)

    def star_thread_code(self):
        self.thread_code=CodeThread(  )
        self.thread_code.start()

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

    def display_image(self,path,label):
        self.clean_display(label)
        image_profile = QtGui.QImage(path) #QImage object
        '''get shape of self.label'''
        width = label.width()
        height = label.height()
        
        '''resize image to fit in self.label'''
        image_profile = image_profile.scaled(width, height, QtCore.Qt.KeepAspectRatio)  
        label.setPixmap(QtGui.QPixmap.fromImage(image_profile)) 
        
        #self.set_progress(progress)

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
        """Put the results in the analyse toolbox"""
        for result in self.result_names:
            self.result_list.addItem(result)

    def change_between_train_and_analyse(self):
        """Change between the train and analyse toolbox"""
        if self.toolBox.currentWidget()==self.train_toolbox:
            self.R_up_frame.show()
            self.R_bot_frame.hide()
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

    '''detect actualization of file in a folder'''
    def run_code_in_thread(self):
        """Run the code in a thread"""
        self.thread = CodeThread(self)
        self.thread.start()

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
        '''star pyqt5 progress bar'''


        self.trainer = ModelTrainer(self.selected_dataset_name,self.params)
        self.trainer.train(self.current_dataset,window,show_plots=False,save_plots=True)
        self.test_model()

    def test_model(self):
        """Test the model in the test and validation dataset"""
        self.trainer.plot_history()
        self.trainer.test_model_with_generator(self.current_dataset,'test')
        self.trainer.test_model_with_generator(self.current_dataset,'val')
        #window.train_progress.setValue(100)

class MyEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == "gui/images\plot.png":
            time.sleep(2)
            MainWindow.display_image(self,'gui/images/plot.png',window.train_label)

'''class to run code in a thread'''
class CodeThread(QThread):
    def __init__(self):
        QThread.__init__(self)
        
    def run(self):
        self.run_code()

    def run_code(self):
        observer = Observer()
        event_handler = MyEventHandler()
        observer.schedule(MyEventHandler(), "gui/images", recursive=False)
        observer.start()
        try:
            while observer.is_alive():
                time.sleep(1)
                observer.join(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())