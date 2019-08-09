import numpy as np

def save_data(path, train_data, train_label, test_data, test_label):
    np.savez(path, 
             alex_train_data=train_data, 
             alex_test_data = test_data, 
            alex_train_label = train_label,
            alex_test_label = test_label)

def load_data(path='alex_mnist_data.npz'):
    """ Loads the Alexnet (256 or 224)mnist dataset"""
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['alex_train_data'], f['alex_train_label']
        x_test, y_test = f['alex_test_data'], f['alex_test_label']
    return (x_train, y_train),(x_test, y_test)
