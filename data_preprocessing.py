import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    # Split datasets
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']

    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label']

    # Normalize pixel values [0, 1]
    X_train_normalized = X_train/ 255.0
    X_test_normalized = X_test / 255.0

    # Feature Scaling (0 mean, 1 SD)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_normalized)
    X_test_scaled = scaler.transform(X_test_normalized)

    return X_train_scaled, y_train, X_test_scaled, y_test

if __name__ == "__main__":
    preprocess()