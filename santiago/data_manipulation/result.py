
from os.path import join
import pandas as pd
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

    def make_results_dataframe(self,checked_experiment_list):
        val_df_list=[]
        test_df_list=[]
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        if len(checked_experiment_list)==0:
            return
        for experiment in checked_experiment_list:
            temp_test_df = pd.read_csv(join(experiment,'metrics_evaluation','test','test_metrics.csv'))
            temp_test_df['experiment']=experiment
            test_df_list.append(temp_test_df)
            temp_val_df = pd.read_csv(join(experiment,'metrics_evaluation','val','val_metrics.csv'))
            temp_val_df['experiment']=experiment
            val_df_list.append(temp_val_df)
        self.val_df = pd.concat(val_df_list)
        self.test_df = pd.concat(test_df_list)

       #print(self.val_df)
            


    
