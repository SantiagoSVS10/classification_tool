from santiago.data_manipulation.dataset import Dataset_initializer
from santiago.utils.json_params import Params
from santiago.model.trainer import ModelTrainer

params_path='data/config/params.json'
params=Params(params_path)

dataset_name='pets'
dataset = Dataset_initializer(dataset_name,params)

dataset.create_distribution(0.8,0.1,0.1)
#dataset.create_new_distribution(0.7,0.2,0.1)
# dataset2.create_distribution(0.5,0.3,0.2)
# dataset3.create_distribution(0.7,0.2,0.1)
# dataset4.create_distribution(0.7,0.2,0.1)

dataset.plot_current_training_distribution()
#dataset.create_new_distribution(0.8,0.1,0.1)
dataset.create_data_generators()

trainer = ModelTrainer(dataset_name,params)

trainer.train(dataset,show_plots=False,save_plots=False)

trainer.test_model_with_generator(dataset,'val')