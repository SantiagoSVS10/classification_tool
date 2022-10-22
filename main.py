from santiago.data_manipulation.dataset import Dataset_initializer

dataset = Dataset_initializer('pets')

dataset2 = Dataset_initializer('enviroments')


#dataset2.create_distribution(0.5,0.3,0.2)
dataset.create_distribution(0.6,0.4,0.0)
dataset.create_new_distribution(0.8,0.1,0.1)



