import math

import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import scipy.sparse as sp


class HCRLDataset:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path

    def preprocess(self, is_save=True, save_df_path=None, save_np_path=None, save_output_path=None):
        if is_save == True and save_df_path is not None:
            self.csv_df = self.load_dataset(self.csv_path)
            with open(save_df_path, "wb") as file:
                pickle.dump(self.csv_df, file)
        
        if is_save == True and save_np_path is not None:
            self.np_array = self.convert_to_numpy(self.csv_df)
            with open(save_np_path, "wb") as file:
                pickle.dump(self.np_array, file)

        if is_save == True and save_output_path is not None:
            self.output_array = self.convert_to_output(self.np_array)
            with open(save_output_path, "wb") as file:
                pickle.dump(self.output_array, file)

    def load_df_file(self, df_path):
        with open(df_path, "rb") as file:
            self.csv_df = pickle.load(file)

    def load_np_file(self, np_path):
        with open(np_path, "rb") as file:
            self.np_array = pickle.load(file)
    
    def load_output_file(self, output_path):
        with open(output_path, "rb") as file:
            self.output_array = pickle.load(file)

    def get_np_array(self):
        return self.np_array
    
    def get_output_array(self):
        return self.output_array

    def get_H(self, series_aid):
        count = series_aid.value_counts()
        p_i = count / series_aid.shape[0]
        return - (p_i * np.log(p_i)).sum()

    def load_dataset(self, path) :
        # parse the dataset
        df = pd.read_csv(path)

        # check the integrity
        assert df.isna().any().any() == False, 'There is at least one missing value.'
        assert df['Timestamp'].is_monotonic_increasing, 'Timestamp is not sorted.'

        # type-cast
        df['abstime'] = pd.to_datetime(df['Timestamp'], unit='s').round('us')
        df['monotime'] = df['Timestamp'] - df['Timestamp' ].min()
        df['aid_int'] = df['Arbitration_ID'].map(lambda x: int(x, 16))
        df['y'] = df['Class'].map({'Normal': 0, 'Attack': 1})

        # calculate the stream-wise timedelta
        df['time_interval'] = df.groupby('Arbitration_ID')['Timestamp'].diff()

        # calculate entropy
        df['entropy'] = df.rolling(window=2402, min_periods=2402, step=10)['aid_int'].apply(self.get_H)
        df['entropy'] = df['entropy'].ffill()

        # process NaN
        df_avg = df.groupby('Arbitration_ID')['time_interval'].mean().rename('avg_time_interval')
        df = df.join(df_avg, on='Arbitration_ID')

        return df
    
    def convert_to_numpy(self, csv_df):
        entropy_avg = csv_df['entropy'].mean()

        np_array = np.zeros((csv_df.shape[0], 15))
        for num1 in tqdm(range(csv_df.shape[0])):
            # add arbitration ID
            np_array[num1][0] = csv_df.iloc[num1, 8]

            # add CAN data
            can_data = csv_df.iloc[num1, 3]
            can_data = can_data.split(' ')
            # change hex to integer
            for num2 in range(csv_df.iloc[num1, 2]):
                np_array[num1][1 + num2] = int(can_data[num2], 16)
            # add 255 to blank space
            for num2 in range(8 - csv_df.iloc[num1, 2]):
                np_array[num1][1 + csv_df.iloc[num1, 2] + num2] = 255
            
            # add time interval
            if pd.isna(csv_df.iloc[num1, 10]) == True:
                np_array[num1][9] = csv_df.iloc[num1, 12]
            else:
                np_array[num1][9] = csv_df.iloc[num1, 10]
            
            # add entropy
            if pd.isna(csv_df.iloc[num1, 11]) == True:
                np_array[num1][10] = entropy_avg
            else:
                np_array[num1][10] = csv_df.iloc[num1, 11]

            # add output
            np_array[num1][14] = csv_df.iloc[num1, 9]

        return np_array
    
    def convert_to_output(self, np_array, num_classes=2):
        output_array = np.zeros((np_array.shape[0], num_classes))
        attack = 0
        normal = 0
        for num1 in range(np_array.shape[0]):
            if np_array[num1][14] == 1:
                output_array[num1][1] = 1
                attack += 1
            else:
                output_array[num1][0] = 1
                normal += 1
        print("Attack: " + str(attack) + ", Normal: " + str(normal))
        return output_array

def row_wise_normalize(mx):
    """Row-wise normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class Graph():
    def __init__(self, num_of_nodes, N= 50, directed=True):
        self.num_of_nodes = num_of_nodes  # Number of nodes
        self.directed = directed
        self.list_of_edges = []  # List of edges

        self.edge_matrix = np.zeros((N, N))  # Adjacent matrix
        self.weight_matrix = np.zeros((N, N))  # Weight matrix

        self.adjacency_list = {node: set() for node in range(num_of_nodes)}

    def add_node(self):

        self.num_of_nodes += 1

    def add_edge(self, node1, node2, weight):
        if self.edge_matrix[node1][node2]:  # If node1 and node2 are connected
            self.weight_matrix[node1][node2] += weight
            self.adjacency_list[node1] = [node1, node2, self.adjacency_list[node1][2] + weight]
        else:  # If node1 and node2 are not connected
            self.edge_matrix[node1][node2] = 1
            self.weight_matrix[node1][node2] = weight
            self.adjacency_list[node1] = [node1, node2, weight]

    def record(self):  # Number of edges, maximum degree, number of nodes
        rec = []
        rec.append(np.sum(self.edge_matrix))
        rec.append(np.max((self.weight_matrix)))
        rec.append(self.num_of_nodes)
        return rec


class TCGInput:
    def __init__(self, window_size):
        self.window_size = window_size

    def set_np_array(self, np_array):
        self.np_array = np_array

    def get_tcg(self, start_idx):
        dic_search = {}
        graph = Graph(0, self.window_size)
        for num1 in range(self.window_size):
            now = self.np_array[start_idx + num1][0]
            if num1 == 0:
                last = now
                continue
            if not (last in dic_search.keys()):
                dic_search[last] = len(dic_search)
                graph.add_node()
            if not (now in dic_search.keys()):
                dic_search[now] = len(dic_search)
                graph.add_node()

            graph.add_edge(dic_search[now], dic_search[last], 1)
            last = now
            
        graph_record = graph.record()
        for num1 in range(self.window_size):
            self.np_array[start_idx + num1][11] = graph_record[0]
            self.np_array[start_idx + num1][12] = graph_record[1]
            self.np_array[start_idx + num1][13] = graph_record[2]
        
        return row_wise_normalize(self.np_array[start_idx: start_idx + self.window_size, 0:14])


class CRG:
    def __init__(self, window_size):
        self.window_size = window_size

    def set_np_array(self, np_array):
        self.np_array = np_array

    def get_crg(self, start_idx):
        crg_arr = np.zeros((self.window_size, self.window_size))
        for num1 in range(self.window_size):
            for num2 in range(self.window_size):
                if self.np_array[num1 + start_idx][0] == self.np_array[num2 + start_idx][0]:
                    crg_arr[num1][num2] = 1
        for num1 in range(self.window_size):
            crg_arr[num1][num1] = 1
        return row_wise_normalize(crg_arr)
    
        

if __name__ == "__main__":
    f_path = "Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv"

    """
    hcrl_dataset1 = HCRLDataset(f_path)
    hcrl_dataset1.preprocess(is_save=True, save_df_path="preprocessed_dataset/d2_df.pkl")

    hcrl_dataset2 = HCRLDataset(f_path)
    hcrl_dataset2.load_df_file("preprocessed_dataset/d2_df.pkl")
    hcrl_dataset2.preprocess(is_save=True, save_np_path="preprocessed_dataset/d2_np.pkl")
    """
    hcrl_dataset3 = HCRLDataset(f_path)
    hcrl_dataset3.load_np_file("preprocessed_dataset/d2_np.pkl")
    hcrl_dataset3.preprocess(is_save=True, save_output_path="preprocessed_dataset/d2_output.pkl")
    print(hcrl_dataset3.output_array)

    """
    hrcl_dataset = HCRLDataset(f_path)
    hrcl_dataset.load_np_file("preprocessed_dataset/d2_np.pkl")

    print(hrcl_dataset.np_array)
    tcg_input = TCGInput(5)
    tcg_input.set_np_array(hrcl_dataset.get_np_array())
    print(tcg_input.get_tcg(12345))
    crg = CRG(5)
    crg.set_np_array(hrcl_dataset.get_np_array())
    print(crg.make_crg(12234))
    """