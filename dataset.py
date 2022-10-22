from os.path import join
import shutil
import math
import os


class Dataset_initializer():
    datasets_path = 'data/datasets'
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path=  join(self.datasets_path,dataset_name)
        self.train_path=join(self.dataset_path,'train')
        self.val_path=join(self.dataset_path,'val')
        self.test_path=join(self.dataset_path,'test')
        self.classes=self.get_classes()
        self.verify_dataset_folders()

    def create_distribution(self,train_percentage=0.8,val_percentage=0.1,test_percentage=0.1):
        """ Create the distribution of the dataset in train, validation and test folders"""

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

        print(f'Dataset {self.dataset_name} prepared!')

    def distribute_images(self):
        """ Copy the images in train, validation and test folders"""
        #print(self.distributed_class_files)
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
                    
        
    def create_train_val_test_folders(self):
        """ Create the folders train, validation and test"""
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

    def get_classes(self):
        listdir=os.listdir(self.dataset_path)
        if os.path.exists(self.train_path) or os.path.exists(self.val_path) or os.path.exists(self.test_path):
            listdir.remove('val')
            listdir.remove('test')
            listdir.remove('train')
        return listdir


    def verify_dataset_folders(self):
        """ Verify if the dataset has the folders train, validation and test"""
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path) or not os.path.exists(self.test_path):
            print(f'Dataset {self.dataset_name} needs to be prepared!')
        