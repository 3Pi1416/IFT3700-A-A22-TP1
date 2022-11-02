from keras.datasets import mnist


#how to import mnist https://www.digitalocean.com/community/tutorials/mnist-dataset-in-python
if __name__ == '__main__':
    (data_to_train_X, data_to_train_y), (data_to_test_X, data_to_test_y) = mnist.load_data()
    print("wow")
