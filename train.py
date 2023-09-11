
from __future__ import division
from __future__ import print_function
from loss import Myloss
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import load_data, accuracy
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import Sizes
import sympy
from models import GAT, SpGAT
from clac_metric import get_metrics
import  pandas as pd
import scipy.sparse as sp
from scipy.linalg import  pinv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
# parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
# parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# parser.add_argument('--patience', type=int, default=100, help='Patience')
#
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)





# Model and optimizer
# if args.sparse:
#     model = SpGAT(nfeat=train_features.shape[1],
#                 nhid=args.hidden,
#                 nclass=8,
#                 dropout=args.dropout,
#                 nheads=args.nb_heads,
#                 alpha=args.alpha)
# else:
#     model = GAT(nfeat=train_features.shape[1],
#                 nhid=args.hidden,
#                 nclass=8,
#                 dropout=args.dropout,
#                 nheads=args.nb_heads,
#                 alpha=args.alpha)
# optimizer = optim.Adam(model.parameters(),
#                          lr=args.lr,
#                         weight_decay=args.weight_decay)

# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
#



def train(epoch, train_data,sizes):
    t = time.time()
    model.train()
    score = model(train_data['feature'], train_data['Adj'],1)
    regression_crit = Myloss()
    loss = regression_crit(train_data['Y_train'] , score,model.drug_l, model.mic_l, model.alpha1,
                               model.alpha2, sizes)

    model.alpha1 = torch.mm(
        torch.mm((torch.mm(model.drug_k, model.drug_k) +model.lambda3*torch.eye(269)).inverse(), model.drug_k),
        2 * train_data['Y_train'] - torch.mm(model.alpha2.T, model.mic_k.T)).detach()
    model.alpha2 = torch.mm(torch.mm((torch.mm(model.mic_k, model.mic_k)  + model.lambda4*torch.eye(598)).inverse(), model.mic_k),
                        2 * train_data['Y_train'].T - torch.mm(model.alpha1.T, model.drug_k.T)).detach()

    # print( torch.mm(model.drug_ps[1]* model.drugs_kernels[1]+ model.drug_ps[2]*model.drugs_kernels[2]+ model.drug_ps[3]*model.drugs_kernels[3],torch.from_numpy(pinv(model.drugs_kernels[0].detach().numpy()))).shape)
    # model.drug_ps[0]=torch.sub(torch.mm(2 * train_data['Y_train']-torch.mm(model.alpha2.T,model.mic_k.T),torch.from_numpy(pinv(torch.mm(model.drugs_kernels[0],model.alpha1).detach().numpy()))),
    #                            torch.mm(model.drug_ps[1]* model.drugs_kernels[1]+ model.drug_ps[2]*model.drugs_kernels[2]+ model.drug_ps[3]*model.drugs_kernels[3],torch.from_numpy(pinv(model.drugs_kernels[0].detach().numpy()))))
    # print(2 * train_data['Y_train']-torch.mm(model.alpha2.T,model.mic_k.T))
    # print(torch.mm(model.drugs_kernels[0],model.alpha1,))mm
    # y=score.detach().numpy()
    # predict_y_proba=[y[idx_test[i][0]-1][idx_test[i][1]-1] for i in range(0,len(idx_test)) ]
    # metric_tmp = get_metrics(test_y,predict_y_proba)
    # fpr, tpr, thresholds = roc_curve(test_y, predict_y_proba)
    # roc_auc = auc(fpr, tpr)
    # print(8888888888888888888888888888888888888)
    # print(roc_auc)
    loss = loss.requires_grad_()
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   # scheduler.step()
    print("epoch : %d, loss:%.2f" % (epoch, loss.item()))
    return loss

def random_index(index_matrix, sizes):
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)
    random.shuffle(random_index)
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp


def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(neg_index_matrix, sizes)
    neg_index = random_index(pos_index_matrix, sizes)
    index = [pos_index[i] + neg_index[i] for i in range(sizes.k_fold)]
    return index




def cross_validation_experiment(drug_mic_matrix, drug_matrix, mic_matrix, sizes):
    index = crossval_index(drug_mic_matrix, sizes)          #生成随机的关联对【药物名 微生物名】
    metric = np.zeros((1, 7))
    for k in range(sizes.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        train_matrix = np.matrix(drug_mic_matrix, copy=True)  # 从类数组对象或数据字符串返回矩阵
        train_matrix[tuple(np.array(index[k]).T)] = 0    #十组数据，将第K组设置成测试机，测试机的数据清零
        trainlist=[]
        train_data={}
        for i in range(sizes.k_fold):
            if i!=k:
                trainlist.extend(index[i])
        train_data['train']=torch.DoubleTensor(trainlist)
        drug_len = 269
        mic_len = 598
        train_data['Y_train'] = torch.DoubleTensor(train_matrix)
        #生成训练组的[270*270]的adj
        image = [[0 for col in range(drug_len)] for row in range(drug_len)]
        a = np.array(image)
        a1 = np.hstack((a, train_matrix))
        image2 = [[0 for col in range(mic_len)] for row in range(mic_len)]
        b = np.array(image2)
        a2 = np.hstack((train_matrix.T, b))
        train_data['Adj']= torch.DoubleTensor(np.vstack((a1, a2)))
        #生成训练组的相似矩阵
        test = np.array(index[k])
        d1 = np.hstack((drug_matrix, train_matrix))
        d2 = np.hstack((train_matrix.T, mic_matrix))
        print(d2.shape)
        print(d1.shape)
        features = np.vstack((d1, d2))
        features = features.astype(float)
        train_features = torch.FloatTensor(features)
        train_data['feature']=train_features
        #训练模型
        for epoch in range(sizes.epochs):
            train(epoch,train_data,sizes)
        #在验证集上验证
        score=model(train_data['feature'], train_data['Adj'],0)
        predict_y_proba = score.reshape(drug_len, mic_len).detach().numpy()
        metric_tmp = get_metrics(drug_mic_matrix[tuple(np.array(index[k]).T)],
                                 predict_y_proba[tuple(np.array(index[k]).T)])
        metric =metric + metric_tmp
    metric = np.array(metric / sizes.k_fold)
    return metric
if __name__ == '__main__':
    drug_sim = pd.read_csv('D:\PYproject\GATL\data\drugsimilarity.csv',header=None)  #读取药物的相似矩阵
    drug_sim=drug_sim.values
    drug_sim=np.array(drug_sim)
    disease_sim =  pd.read_csv('D:\PYproject\GATL\data\diseasesimilarity.csv', header=None)  #读取微生物的相似矩阵
    disease_sim=disease_sim.values
    disease_sim=np.array(disease_sim)
    adj_triple = pd.read_csv('D:\PYproject\GATL\data\\adj.csv', header=None)       #读取关联信息，创建行宽药物，列宽微生物的关联矩阵
    adj_triple = adj_triple.values
    adj_triple = np.array(adj_triple)
    drug_pr = pd.read_csv('D:\PYproject\GATL\data\drug269_protein(quan).csv', header=None)  # 读取药物的相似矩阵
    drug_pr = drug_pr.values
    drug_pr = np.array(drug_pr)
    di_pr = pd.read_csv('D:\PYproject\GATL\data\disease269_protein(quan).csv', header=None)  # 读取药物的相似矩阵
    di_pr = di_pr.values
    di_pr = np.array(di_pr)

    drug_pr_matrix = sp.csc_matrix((drug_pr[:, 2], (drug_pr[:, 0], drug_pr[:, 1])),shape=(269, 13271)).toarray()
    mic_pr_matrix = sp.csc_matrix((mic_pr[:, 2], ( mic_pr[:, 0],  mic_pr[:, 1])),
                                   shape=(len(disease_sim), 13271)).toarray()

    drug_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0], adj_triple[:, 1])),
                                    shape=(len(drug_sim), len(disease_sim))).toarray()

    sizes = Sizes(drug_sim.shape[0], disease_sim.shape[0])
    drug_sim = np.hstack((drug_sim, drug_pr_matrix ))
    disease_sim = np.hstack((mic_pr_matrix, disease_sim))
    # print(drug_sim.shape)
    # print(mic_sim.shape)
    model = GAT(nfeat=14138,
                                nhid=sizes.hidden,
                                nclass=sizes.nclass,
                                dropout=sizes.dropout,
                                nheads=sizes.nb_heads,
                                alpha=sizes.alpha)
    optimizer = optim.Adam(model.parameters(),
                            lr=sizes.lr,
                            weight_decay=sizes.weight_decay)
    result= cross_validation_experiment(drug_mic_matrix, drug_sim, disease_sim, sizes)



