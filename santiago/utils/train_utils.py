from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
plt.ioff()

from IPython.display import clear_output
import tensorflow as tf
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

class PlotLearning(tf.keras.callbacks.Callback):
    def __init__(self):
        plt.rcParams["figure.figsize"]=15,7
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();

class SavePlotLearning(tf.keras.callbacks.Callback):
    def __init__(self,window):
        self.window = window
        self.epochs = self.window.params.num_epochs
        self.progress = 0
        plt.rcParams["figure.figsize"]=(9,4.5)
        self.savepath = 'gui/images/plot.png'
        '''delete white borders from matplotlib plots'''

    def on_train_begin(self, logs={}):
        #self.window.train_progress.setValue(0)
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.savefig(self.savepath,bbox_inches='tight')
        plt.close()
        self.window.display_image(self.savepath,self.window.train_label)

    def on_epoch_end(self, epoch,logs={}):
        self.progress+=100/self.epochs

        '''update progress bar in main thread'''
        QtCore.QMetaObject.invokeMethod(self.window.train_progress, "setValue", QtCore.Qt.QueuedConnection,
                                        QtCore.Q_ARG(int, self.progress))
        #self.window.train_progress.setValue(self.progress)
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
    
        #plt.show();
        plt.savefig(self.savepath,bbox_inches='tight')
        plt.close()
        self.window.display_image(self.savepath,self.window.train_label)
        

   