import os
import pandas as pd
from  CLI.data_processing.dataset_split import create_subsets
from CLI.data_processing.features import create_feature_matrix
from Core.neural_network.neural_network import Network

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # points to AutoPredict/programs
TRAIN_SET_PATH = os.path.join(BASE_DIR, '..', 'Data', 'Processed Data', 'Subsets', 'training_set.csv')
CV_SET_PATH = os.path.join(BASE_DIR, '..', 'Data', 'Processed Data', 'Subsets', 'cv_set.csv')
TEST_SET_PATH = os.path.join(BASE_DIR, '..', 'Data', 'Processed Data', 'Subsets', 'test_set.csv')

def pick_file_from_folder():
    DATASET_PATH = os.path.join(BASE_DIR, '..', 'Data', 'Processed Data', 'Modified Raw Data')  # go up two levels, then into datasets
    DATASET_PATH = os.path.abspath(DATASET_PATH)  # normalize path

    files = [f for f in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, f))]

    print("Select a file:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    choice = int(input("Enter the file number: "))
    selected_file = os.path.join(DATASET_PATH, files[choice])
    return selected_file

def data_split(data_file, seed=69):
    create_subsets(data_filename=data_file, training_filename=TRAIN_SET_PATH, cv_filename=CV_SET_PATH, test_filename=TEST_SET_PATH, seed=seed)

def create_net(layers, train_set_file=TRAIN_SET_PATH):
    X, y = create_feature_matrix(train_set_file, 'salary_in_usd')
    return Network(layers=layers, X=X, Y=y)

def network_predict(network, cv_file=CV_SET_PATH):
    X, y = create_feature_matrix(cv_file, 'salary_in_usd')
    network.X = X
    network.Y = y

    return network.compute_cost()


