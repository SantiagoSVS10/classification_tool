
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from santiago.data_manipulation.data_generator import DataGenerator
from os.path import join

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import math
import os

'''This class is used to manage the datasets.
Allows to distribute the images to create the train, validation and test data generators.
Plots the distribution of the images in the datasets.
Create all the necessary folders to save the results.'''

class Dataset_initializer():
    datasets_path = 'data/datasets'
    results_path = 'data/results'
    gui_images_path = 'gui/images'
    def __init__(self, dataset_name, params):
        self.params = params
        self.dataset_name = dataset_name
        self.dataset_path=  join(self.datasets_path,dataset_name)
        
        self.train_path=join(self.dataset_path,'train')
        self.val_path=join(self.dataset_path,'val')
        self.test_path=join(self.dataset_path,'test')
        self.train_dataframe=pd.DataFrame()
        self.val_dataframe=pd.DataFrame()
        self.test_dataframe=pd.DataFrame()

        self.classes=self.get_classes()
        self.distributed=self.verify_distribution()
        
        
    '''create initial distribution for datasets with train,test and val folders'''
    def create_distribution(self,train_percentage=0.8,val_percentage=0.1,test_percentage=0.1):

        if math.fsum([train_percentage,val_percentage,test_percentage])!=1.0:
            print('The percentages must sum 1')
            return

        self.distributed_class_files = {}
        for class_name in self.classes:
            class_path = join(self.dataset_path,class_name)
            files_in_class = len(os.listdir(class_path))
            train_images_count=math.floor(files_in_class*train_percentage)
            val_images_count=math.floor(files_in_class*val_percentage)
            test_images_count=math.floor(files_in_class*test_percentage)
            residue=files_in_class-(train_images_count+val_images_count+test_images_count)
            test_images_count+=residue      #add residue to test set
            self.distributed_class_files[class_name] = list([files_in_class,train_images_count,val_images_count,test_images_count])

        self.create_train_val_test_folders()
        self.create_class_folders()
        self.distribute_images()
        self.distributed=True
        print(f'Dataset {self.dataset_name} prepared!')

    '''create new distribution for dataframe with classes folders, using dataframe'''
    def create_distribution_dataframe(self,train_percentage=0.7,val_percentage=0.2,test_percentage=0.1):
        
        if math.fsum([train_percentage,val_percentage,test_percentage])!=1.0:
            print('The percentage must sum 1')
            return

        for classe in self.classes:
            '''get filenames of each class'''
            class_path = join(self.dataset_path,classe)
            files_in_class = os.listdir(class_path)
            '''convert to absolute path'''
            files_in_class = [join(class_path,file) for file in files_in_class]
            '''create dataframe of each class'''
            class_dataframe = pd.DataFrame(files_in_class,columns=['filename'])
            class_dataframe['class']=classe
            '''add dataframe to the dataset dataframe'''
            self.train_dataframe = self.train_dataframe.append(class_dataframe[:math.floor(len(files_in_class)*train_percentage)])
            self.val_dataframe = self.val_dataframe.append(class_dataframe[math.floor(len(files_in_class)*train_percentage):math.floor(len(files_in_class)*(train_percentage+val_percentage))])
            self.test_dataframe = self.test_dataframe.append(class_dataframe[math.floor(len(files_in_class)*(train_percentage+val_percentage)):])
        self.distributed='dataframe'
        print(f'Dataset {self.dataset_name} prepared by dataframe!')

    '''create new distribution for datasets with train,test and val folders'''
    def create_new_distribution(self,train_percentage=0.8,val_percentage=0.1,test_percentage=0.1):
        """ Create a new distribution of the dataset in train, validation and test folders"""
        if math.fsum([train_percentage,val_percentage,test_percentage])!=1.0:
            print('The percentages must sum 1')
            return
        
        if self.distributed=='folders':

            '''delete the old distribution'''
            try:
                shutil.rmtree(self.train_path)
                shutil.rmtree(self.val_path)
                shutil.rmtree(self.test_path)
            except:
                pass
            '''create the new distribution'''
            print('Creating new distribution..')
            self.create_distribution(train_percentage,val_percentage,test_percentage)

        if self.distributed=='dataframe':
            '''erase the old distribution'''
            self.train_dataframe=pd.DataFrame()
            self.val_dataframe=pd.DataFrame()
            self.test_dataframe=pd.DataFrame()
            self.create_distribution_dataframe(train_percentage,val_percentage,test_percentage)
            

    '''distrubute images of classes in train, val and test folders'''
    def distribute_images(self):
        """ Copy the images in train, validation and test folders"""
        print(self.distributed_class_files)
        for class_name in self.distributed_class_files:
            '''get filenames of each class'''
            class_path = join(self.dataset_path,class_name)
            files_in_class = os.listdir(class_path)
            '''get the number of images to be distributed'''
            train_images = self.distributed_class_files[class_name][1]
            val_images = self.distributed_class_files[class_name][2]
            test_images = self.distributed_class_files[class_name][3]
            '''distribute the images'''
            for i in range(train_images):
                if not os.path.exists(join(self.train_path,class_name,files_in_class[i])): 
                    shutil.copy(join(class_path,files_in_class[i]),join(self.train_path,class_name,files_in_class[i]))
            for i in range(train_images,train_images+val_images):
                if not os.path.exists(join(self.val_path,class_name,files_in_class[i])): 
                    shutil.copy(join(class_path,files_in_class[i]),join(self.val_path,class_name,files_in_class[i]))
            for i in range(train_images+val_images,train_images+val_images+test_images):
                if not os.path.exists(join(self.test_path,class_name,files_in_class[i])): 
                    shutil.copy(join(class_path,files_in_class[i]),join(self.test_path,class_name,files_in_class[i]))
            
    '''create folder for each class in train, val and test folders'''
    def create_class_folders(self):
        """ Create folders of each class inside train, validation and test"""
        for class_name in self.distributed_class_files:
            class_train_path = join(self.train_path,class_name)
            class_val_path = join(self.val_path,class_name)
            class_test_path = join(self.test_path,class_name)
            if not os.path.exists(class_train_path):
                os.mkdir(class_train_path)
            if not os.path.exists(class_val_path):
                os.mkdir(class_val_path)
            if not os.path.exists(class_test_path):
                os.mkdir(class_test_path)
                    
    '''create folders for train, validation and test'''
    def create_train_val_test_folders(self):
        """ Create the folders train, validation and test"""
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

    '''get the classes considering the folders of dataset'''
    def get_classes(self):
        
        if os.path.exists(self.train_path):
            listdir=os.listdir(join(self.dataset_path,'train'))
        else:
            listdir=os.listdir(self.dataset_path)
        print(listdir)
        return listdir


    def verify_distribution(self):
        """ Verify if the dataset has the folders train, validation and test"""
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path) or not os.path.exists(self.test_path):
            print(f'Dataset {self.dataset_name} needs to be prepared!')
            self.create_results_folder()
            return 'dataframe'
        else:
            print(f'Dataset {self.dataset_name} is ready!')
            self.create_results_folder()
            return 'folders'

    def create_results_folder(self):
        '''create a folder with the name of current dataset inside results path'''
        self.results_dataset_path = join(self.results_path,self.dataset_name)
        if not os.path.exists(self.results_dataset_path):
            os.mkdir(self.results_dataset_path)
        else:
            print('Results folder already exists!')

    
    def plot_current_training_distribution_dataframe(self):
        """ Plot the current distribution of the train, validation and test dataframes"""
        count=[]
        for class_name in self.classes:
            class_df = self.train_dataframe[self.train_dataframe['class']==class_name]
            count.append(len(class_df))
        for class_name in self.classes:
            class_df = self.val_dataframe[self.val_dataframe['class']==class_name]
            count.append(len(class_df))
        for class_name in self.classes:
            class_df = self.test_dataframe[self.test_dataframe['class']==class_name]
            count.append(len(class_df))

        classe=[]
        classe=self.classes*3  
        # print(set)
        # print(count)
        # print(classe)
        # print(len(set),len(count),len(classe))
        set=['train']*len(self.classes)+['val']*len(self.classes)+['test']*len(self.classes)
        distribution_df = pd.DataFrame({'set':set,'count':count,'class':classe})
        # print(distribution_df)
        sns.set(rc={'figure.figsize':(9,4.5)})
        barplot=sns.barplot(x='set', y='count', hue='class', data=distribution_df)
        plt.title('Training distribution')
        '''change font size of plot'''
        for item in ([barplot.title, barplot.xaxis.label, barplot.yaxis.label] +
                barplot.get_xticklabels() + barplot.get_yticklabels()):
            item.set_fontsize(13)

        barplot.figure.savefig(join(self.gui_images_path,'current_training_distribution.png'),bbox_inches='tight')
        plt.close()
        return barplot
        


    def plot_current_training_distribution(self):
        """ Plot the current distribution of the dataset in train, validation and test folders"""
        
        set=['train']*len(self.classes)
        set.extend(['val']*len(self.classes))
        set.extend(['test']*len(self.classes))
        
        count=[]
        for class_name in self.classes:
            count.append(len(os.listdir(join(self.train_path,class_name))))
        for class_name in self.classes:
            count.append(len(os.listdir(join(self.val_path,class_name))))
        for class_name in self.classes:
            count.append(len(os.listdir(join(self.test_path,class_name))))
        print(count)
        classe=[]
        classe=self.classes*3  
        # print(set)
        # print(count)
        # print(classe)
        # print(len(set),len(count),len(classe))
        distribution_df = pd.DataFrame({'set':set,'count':count,'class':classe})
        print(distribution_df)

        sns.set(rc={'figure.figsize':(9,4.5)})
        barplot=sns.barplot(x='set', y='count', hue='class', data=distribution_df)
        plt.title('Training distribution')
        '''change font size of plot'''
        for item in ([barplot.title, barplot.xaxis.label, barplot.yaxis.label] +
                barplot.get_xticklabels() + barplot.get_yticklabels()):
            item.set_fontsize(13)

        barplot.figure.savefig(join(self.gui_images_path,'current_training_distribution.png'),bbox_inches='tight')
        plt.close()
        return barplot

    def verify_images_integrity(self):
        """ Verify if the images are corrupted"""
        for class_name in self.classes:
            for set in ['train','val','test']:
                for image in os.listdir(join(self.dataset_path,set,class_name)):
                    try:
                        #img_bytes = tf.io.read_file(join(self.dataset_path,set,class_name,image))
                        #decoded_img = tf.decode_image(img_bytes)
                        print(f'Image {image} is ok!')
                    except:
                        '''delete corrupted image'''
                        os.remove(join(self.dataset_path,set,class_name,image))
                        print(f'Image {image} is corrupted and was deleted!')


    '''function to create generator from dataframe'''
    @staticmethod
    def generator_dataframe(params,batch_size=1, df=None, shuffle=False):
        gen = ImageDataGenerator(rescale=1./255)
        """ Create a generator from dataframe"""
        generator = gen.flow_from_dataframe(
            x_col='filename',
            y_col='class',
            dataframe=df,
            # directory="train",
            class_mode='binary',
            # classes=None,
            color_mode='rgb',
            batch_size=batch_size,
            target_size=(params.image_height,params.image_width),
            shuffle=shuffle,
            seed=123)

        return generator

    @staticmethod
    def generate(path,params,shuffle=False):
        image_size = (params.image_height,params.image_width)
        #generated = tf.keras.preprocessing.image_dataset_from_directory(
        generated= tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
        path,
        #labels="inferred",
        #label_mode="binary", #categorical
        #class_names=classes,
        #image_size=image_size,
        shuffle=shuffle,
        seed=123,
        batch_size=params.batch_size,

        target_size=image_size,
        class_mode='binary',
    )
        return generated
        
    def create_data_generators(self):
        """ Create the data generators for train, validation and test"""
        #self.verify_images_integrity()
        if self.distributed=='folders':
            self.train_generator = self.generate(self.train_path,self.params,shuffle=True)
            
            self.val_generator = self.generate(self.val_path,self.params)
            
            self.test_generator = self.generate(self.test_path,self.params)

        if self.distributed=='dataframe':
            self.train_generator = self.generator_dataframe(self.params, df=self.train_dataframe, shuffle=True)
            print(self.train_generator.filepaths)
            self.val_generator = self.generator_dataframe(self.params, df=self.val_dataframe)

            self.test_generator = self.generator_dataframe(self.params, df=self.test_dataframe)

        print('Data generators created!')
