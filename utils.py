import math

import numpy as np
import scipy.sparse as sp
import torch
import  pandas as pd

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(data_path="F:\PYproject\GAT\data", dataset="\MY"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    drug_sim = np.loadtxt(data_path + dataset+ '\drugsimilarity.txt', delimiter='\t')
    mic_sim = np.loadtxt(data_path + dataset+ '\microbesimilarity.txt', delimiter='\t')

    # [0,adj] [adj.T,0]拼接
    adj_triple = pd.read_csv('F:\PYproject\GATL\data\MY\\adj.csv', header=None)
    adj_triple = adj_triple.values
    adj_triple = np.array(adj_triple)
    adj = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0]-1, adj_triple[:, 1]-1)), shape=(len(drug_sim),len(mic_sim))).toarray()
    train_y=adj
    image = [[0 for col in range(175)] for row in range(175)]
    a = np.array(image)
    a1 = np.hstack((a, adj))
    image2 = [[0 for col in range(95)] for row in range(95)]
    b= np.array(image2)
    a2 = np.hstack((adj.T,b))
    adj=np.vstack((a1,a2))
    # adj += adj.T - np.diag(adj.diagonal())
    adj = adj.astype(float)
    train_adj= torch.FloatTensor(adj)
    train_y = train_y.astype(float)
    train_y = torch.FloatTensor(train_y)



    lin= pd.read_csv('F:\PYproject\GATL\data\MY\\train.csv', header=None)
    lin = lin.values
    lin = np.array(lin)
    lin = sp.csc_matrix((lin[:, 2], (lin[:, 0] - 1, lin[:, 1] - 1)),
                        shape=(len(drug_sim),len(mic_sim))).toarray()
    d=pd.read_csv(data_path + dataset + '\drugsimilarity.csv', header=None)
    d=d.values
    d=np.array(d)
    train_dfeature = np.zeros(d.shape)
    for i in range(0, len(lin)):
        x = lin[i][0] - 1
        train_dfeature[x] = d[x]
    s=pd.read_csv(data_path + dataset + '\microbesimilarity.csv', header=None)
    s=s.values
    s=np.array(s)
    train_mfeature = np.zeros(s.shape)
    for i in range(0, len(lin)):
        x = lin[i][1] - 1
        train_mfeature[x] = s[x]
    d1=np.hstack((train_dfeature,lin))
    d2=np.hstack((lin.T, train_mfeature))
    features = np.vstack((d1,d2))
    features = features.astype(float)
    train_features = torch.FloatTensor(features)




    #文件中，1表示0，2表示1
    idx_test=pd.read_csv('F:\PYproject\GATL\data\MY\\test2.csv', header=None)
    idx_test=idx_test.values
    idx_test=np.array(idx_test)
    test_y =  [idx_test[i][2]-1 for i in range(0,len(idx_test))]



    return train_adj, train_features,len(drug_sim),len(mic_sim),idx_test,test_y,train_y

def NegativeGenerate(DrugDisease, AllDurg,AllDisease):
    import random
    NegativeSample = []
    counterN = 0
    while counterN < len(DrugDisease):
        counterR = random.randint(0, len(AllDurg) - 1)
        counterD = random.randint(0, len(AllDisease) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllDurg[counterR])
        DiseaseAndRnaPair.append(AllDisease[counterD])
        flag1 = 0
        counter = 0
        while counter < len(DrugDisease):
            if DiseaseAndRnaPair == DrugDisease[counter]:
                flag1 = 1
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample):
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = []
            NamePair.append(AllDurg[counterR])
            NamePair.append(AllDisease[counterD])
            NegativeSample.append(NamePair)
            counterN = counterN + 1
    return NegativeSample



def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = torch.zeros([row, col])
    for i in range(row):
        if max(embeddings[i, :]) == min(embeddings[i, :]):
            ne[i, :] = embeddings[i, :]
        else:
            ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))

    return ne


def getGipKernel(y, trans, gamma, normalized=False):

    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)

    krnl = torch.mm(y, y.T)

    diag=torch.diag(krnl)
    mean=torch.nanmean(diag)
    krnl = krnl / mean
    krnl = torch.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    di = torch.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d

def laplacian(kernel):


    d1=torch.nansum(kernel,dim=1)
    D_1 = torch.diag(d1)
    L_D_1 = D_1 - kernel

    D_5 = D_1.rsqrt()
    D_5 = torch.where(torch.isinf(D_5), torch.full_like(D_5, 0), D_5)

    L_D_11 = torch.mm(D_5, L_D_1)

    L_D_11 = torch.mm(L_D_11, D_5)
    return L_D_11



class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.drug_size = drug_size
        self.mic_size = mic_size
        self.k_fold = 5
        self.epochs = 2
        self.seed = 1
        self.hidden=8
        self.dropout=0.8
        self.nb_heads=1
        self.alpha=0.3
        self.lr=0.001
        self.weight_decay=0.03
        self.nclass=8

        self.lambda1 = 2 ** (-4)
        self.lambda2 = 2 ** (-4)
        self.lambda3 = 2 ** (-4)
        self.lambda4 = 2 ** (-4)
