
from os.path import join
import os
class Results():
    results_path = 'data/results'
    def __init__(self,result_name):
        self.result_name = result_name
        print(join(self.results_path,result_name))
        pass

    def get_results_folders(self):
        '''get the folders inside results path'''
        self.dir_list = os.listdir(join(self.results_path,self.result_name))
        return self.dir_list
    
