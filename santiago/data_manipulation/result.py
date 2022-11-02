
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
        self.all_experiment_list = os.listdir(join(self.results_path,self.result_name))
        '''verify if the experiment is completed'''
        self.valid_experiment_list = []
        for experiment in self.all_experiment_list:
            experiment_path=join(self.results_path,self.result_name,experiment)
            if os.path.exists(join(experiment_path,'best_model','best_model.h5')):
                    if os.path.exists(join(experiment_path,'metrics_evaluation','val','val_metrics.csv')):
                        self.valid_experiment_list.append(experiment)
    
