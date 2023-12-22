import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from preprocess import *

def run_preprocess(save_paths, csv_paths):
    for idx in range(len(csv_paths)):
        csv_path = csv_paths[idx]
        save_path = save_paths[idx]
        hcrl_dataset = HCRLDataset(csv_path)
        hcrl_dataset.preprocess(is_save=True, save_df_path=save_path[0], save_np_path=save_path[1], save_output_path=save_path[2])

class CarCANDataset(Dataset):
    def __init__(self, file_paths, window_size):
        self.np_array_list = []
        self.output_array_list = []

        for file_path in file_paths:
            hcrl_dataset = HCRLDataset()
            hcrl_dataset.load_np_file(file_path[0])
            hcrl_dataset.load_output_file(file_path[1])
            self.np_array_list.append(hcrl_dataset.get_np_array())
            self.output_array_list.append(hcrl_dataset.get_output_array())
        

        self.x = []
        self.adj = []
        self.y = []
        self.data_len = 0
        
        for file_num in range(len(file_paths)):
            tcg_input = TCGInput(window_size)
            tcg_input.set_np_array(self.np_array_list[file_num])
            crg = CRG(window_size)
            crg.set_np_array(self.np_array_list[file_num])

            start_idx = 0
            for start_idx in range(0, self.np_array_list[file_num].shape[0] - window_size + 1, window_size):
                self.x.append(tcg_input.get_tcg(start_idx))
                self.adj.append(crg.get_crg(start_idx))
                self.y.append(self.output_array_list[file_num][start_idx:start_idx + window_size, :])

        self.x = torch.FloatTensor(np.array(self.x))
        self.adj = torch.FloatTensor(np.array(self.adj))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.adj[idx], self.y[idx]

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # Matrix multiplication  XW
        output = torch.spmm(adj, support)  # Sparse matrix multiplication AXW
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class NN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(NN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def get_confusion_matrix(y, output):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for num1 in range(len(y)):
        if y[num1] == 0 and output[num1] == 0:
            TN += 1
        if y[num1] == 0 and output[num1] == 1:
            FP += 1
        if y[num1] == 1 and output[num1] == 0:
            FN += 1
        if y[num1] == 1 and output[num1] == 1:
            TP += 1
    return [TN, FP, FN, TP]

def evaluate(TN, FP, FN, TP):
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    print("TP: " + str(TP))
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("F1 score: " + str(f1_score))

def train(dataloader_train, dataloader_test, optimizer, model, window_size, device="cpu"):
    epochs = 50
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        TN = 0
        FP = 0
        FN = 0
        TP = 0
        for samples in tqdm(dataloader_train):
            x, adj, y = samples
            x = x[0].to(device)
            y = y[0].to(device)
            y = torch.argmax(y, dim = 1)
            adj = adj[0].to(device)

            #print(x.shape)
            #print(y.shape)
            #print(adj.shape)

            optimizer.zero_grad()
            output = model.forward(x, adj)

            loss_train = F.nll_loss(output, y)

            loss_train.backward()
            optimizer.step()
            output = torch.argmax(output, dim = 1)
            conf_mat = get_confusion_matrix(y, output)
            TN += conf_mat[0]
            FP += conf_mat[1]
            FN += conf_mat[2]
            TP += conf_mat[3]
        
        print("Train: ", str(epoch))
        evaluate(TN, FP, FN, TP)

        if epoch % 10 == 9:
            model.eval()
            TN = 0
            FP = 0
            FN = 0
            TP = 0
            for samples in tqdm(dataloader_test):
                x, adj, y = samples
                x = x[0].to(device)
                y = y[0].to(device)
                y = torch.argmax(y, dim = 1)
                adj = adj[0].to(device)

                output = model.forward(x, adj)
                output = torch.argmax(output, dim=1)
                conf_mat = get_confusion_matrix(y, output)
                TN += conf_mat[0]
                FP += conf_mat[1]
                FN += conf_mat[2]
                TP += conf_mat[3]
            
            print("Test: ", str(epoch))
            evaluate(TN, FP, FN, TP)
            torch.save(model, "./model_save/" + str(window_size) + "_" + str(epoch) + ".pt")


train_pp_paths = [["./preprocessed_dataset/d1_df.pkl", "./preprocessed_dataset/d1_np.pkl", "./preprocessed_dataset/d1_output.pkl"],
                   ["./preprocessed_dataset/d2_df.pkl", "./preprocessed_dataset/d2_np.pkl", "./preprocessed_dataset/d2_output.pkl"]]

train_csv_paths = ["Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv",
                   "Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_2.csv"]

train_file_paths = [["./preprocessed_dataset/d1_np.pkl", "./preprocessed_dataset/d1_output.pkl"],
                    ["./preprocessed_dataset/d2_np.pkl", "./preprocessed_dataset/d2_output.pkl"],]

test_pp_paths = [["./preprocessed_dataset/sd_df.pkl", "./preprocessed_dataset/sd_np.pkl", "./preprocessed_dataset/sd_output.pkl"]]

test_csv_paths = ["Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/1_Submission/Pre_submit_D.csv"]

test_file_paths = [["./preprocessed_dataset/sd_np.pkl", "./preprocessed_dataset/sd_output.pkl"]]

model = NN(nfeat=14, nhid=32, nclass=2, dropout=0.5).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# run_preprocess(train_pp_paths, train_csv_paths)
# run_preprocess(test_pp_paths, test_csv_paths)
car_can_dataset_train = CarCANDataset(train_file_paths, 50)
car_can_dataloader_train = DataLoader(car_can_dataset_train, batch_size=1, shuffle=True)
car_can_dataset_test = CarCANDataset(test_file_paths, 50)
car_can_dataloader_test = DataLoader(car_can_dataset_test, batch_size=1, shuffle=False)

train(car_can_dataloader_train, car_can_dataloader_test, optimizer, model, 50, "cuda")
