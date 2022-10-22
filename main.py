from santiago.data_manipulation.dataset import Dataset_initializer

dataset = Dataset_initializer('pets')

dataset2 = Dataset_initializer('enviroments')


dataset2.create_distribution(0.7,0.2,0.1)
dataset.create_distribution(0.7,0.2,0.1)


