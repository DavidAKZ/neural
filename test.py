'''
Created on 4 Jan 2022

@author: davidklemitz
'''
import mnist_loader
import simple_nn

training_data, validation_data, test_data = mnist_loader.load_data_together()


net = simple_nn.Network([784, 30, 10])

net.SGD(training_data, 10, 10, 3.0, test_data = test_data)

