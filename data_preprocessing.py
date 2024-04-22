import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def preprocess():
    # Data paths
    train_data_file = 'sign_mnist_train.csv'
    test_data_file = 'sign_mnist_test.csv'
    folder_name = 'Project Data'

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    train_data_path = os.path.join(desktop_path, folder_name, train_data_file)
    test_data_path = os.path.join(desktop_path, folder_name, test_data_file)

    # Load data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_labels = train_df['label']
    train = train_df.drop(['label'], axis=1)

    test_labels = test_df['label']
    test = test_df.drop(['label'], axis=1)

    # Reshaping images
    train_images = train.values
    train_images = np.array([np.reshape(i, (28, 28)) for i in train_images])
    train_images = np.array([i.flatten() for i in train_images])

    test_images = test.values
    test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
    test_images = np.array([i.flatten() for i in test_images])

    # One hot encoding labels
    binrizer = LabelBinarizer()
    train_labels = binrizer.fit_transform(train_labels)
    test_labels = binrizer.fit_transform(test_labels)

    # Split into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # Normalize pixel values [0, 1]
    X_train = X_train/ 255.0
    X_valid = X_valid / 255.0

    # Reshape to 4D array (CNN)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    return X_train, X_valid, test_images, y_train, y_valid, test_labels

if __name__ == "__main__":
    preprocess()