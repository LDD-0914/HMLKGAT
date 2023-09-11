import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.alpha=alpha
        #自己的
        # self.alpha3= nn.Parameter(torch.empty(size=(64, 8)))
        # nn.init.xavier_uniform_(self.alpha3.data, gain=1.414)
        self.drug_l = []
        self.mic_l = []
        self.drug_k = []
        self.mic_k = []
        self.drug_size = 269
        self.mic_size = 598
        self.f1=128
        self.f2=24
        self.h1_gamma = 2 ** (-5)
        self.h2_gamma = 2 ** (-3)
        self.h3_gamma = 2 ** (-3)
        self.h4_gamma = 2 ** (-3)

        self.lambda1 = 2 ** (-3)
        self.lambda2 = 2 ** (-4)
        self.lambda3 = 3 ** (-4)
        self.lambda4 = 3 ** (-4)
        self.kernel_len = 3

        # self.drug_ps = torch.tensor([0.35,0.23,0.22,0.2])
        # self.mic_ps =torch.tensor([0.35,0.23,0.22,0.2])
        self.drug_ps = torch.ones(self.kernel_len) / self.kernel_len
        self.mic_ps = torch.ones(self.kernel_len) / self.kernel_len
        self.alpha1 = torch.rand(size=(269, 598)).double()
        # nn.init.xavier_uniform_(self.alpha1.data, gain=1.414)
        self.alpha2 =torch.rand(size=(598, 269)).double()
        # nn.init.xavier_uniform_(self.alpha2.data, gain=1.414)

        self.gat = GraphAttentionLayer(nfeat, self.f1, dropout=dropout, alpha=alpha, concat=False)
        self.attentions = [GraphAttentionLayer(self.f1, self.f2, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(self.f2 * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)




    def forward(self, x, adj,true):

        self.drugs_kernels = []
        self.mic_kernels = []
        x1 = self.gat(x,adj)
        #
        # self.drugs_kernels.append(torch.DoubleTensor(cosine_similarity(x1[:self.drug_size].detach().numpy())))
        # self.mic_kernels.append(torch.DoubleTensor(cosine_similarity(x1[self.drug_size:].detach().numpy())))
        self.drugs_kernels.append(torch.DoubleTensor(getGipKernel(x1[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
        self.mic_kernels.append(torch.DoubleTensor(getGipKernel(x1[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))

        x2 = torch.cat([att(x1, adj) for att in self.attentions], dim=1)
        self.drugs_kernels.append(torch.DoubleTensor(getGipKernel(x2[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
        self.mic_kernels.append(torch.DoubleTensor(getGipKernel(x2[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))
        # self.drugs_kernels.append(torch.DoubleTensor(cosine_similarity(x1[:self.drug_size].detach().numpy())))
        # self.mic_kernels.append(torch.DoubleTensor(cosine_similarity(x1[self.drug_size:].detach().numpy())))
        #
        x3= self.out_att(x2, adj)
        # self.drugs_kernels.append(
        #     torch.DoubleTensor(getGipKernel(x3[:self.drug_size].clone(), 0, self.h1_gamma, True).double()))
        # self.mic_kernels.append(
        #     torch.DoubleTensor(getGipKernel(x3[self.drug_size:].clone(), 0, self.h1_gamma, True).double()))
        self.drugs_kernels.append(torch.DoubleTensor(cosine_similarity(x3[:self.drug_size].detach().numpy())))
        self.mic_kernels.append(torch.DoubleTensor(cosine_similarity(x3[self.drug_size:].detach().numpy())))
        # # # #
        # x4=pd.read_csv('F:\PYproject\GATL\data\me\deepwalkfeature.csv', header=None)
        # x4=x4.values
        # x4=np.array(x4)
        # x4=torch.FloatTensor(x4)
        # drugs_kernels.append(torch.DoubleTensor(getGipKernel(x4[:self.drug_size].clone(), 0, self.h3_gamma, True).double()))
        # mic_kernels.append(torch.DoubleTensor(getGipKernel(x4[self.drug_size:].clone(), 0, self.h3_gamma, True).double()))

        # drug = pd.read_csv('H:\PYproject\GATL\data\me\drugsimilarity.csv', header=None)  # 读取药物的相似矩阵
        # drug = drug.values
        # drug= np.array(drug)
        # mic = pd.read_csv('H:\PYproject\GATL\data\me\diseasesimilarity.csv', header=None)  # 读取微生物的相似矩阵
        # mic = mic.values
        # mic = np.array(mic)
        # drug = torch.DoubleTensor(drug)
        # mic = torch.DoubleTensor(mic)
        # self.drugs_kernels.append(drug)
        # self.mic_kernels.append(mic)


        self.drug_k = sum([self.drug_ps[i] * self.drugs_kernels[i] for i in range(self.kernel_len )])
        self.mic_k = sum([self.mic_ps[i] * self.mic_kernels[i] for i in range(self.kernel_len )])
        self.drug_l = laplacian(self.drug_k)
        self.mic_l = laplacian(self.mic_k)

        out1 = torch.mm(self.drug_k, self.alpha1)
        out2 = torch.mm(self.mic_k, self.alpha2)
        score = (out1 + out2.T) / 2

        return score



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.drug_k = []
        self.mic_k = []
        self.alpha1 = nn.Parameter(torch.empty(size=(nclass, nclass)))
        nn.init.xavier_uniform_(self.alpha1.data, gain=1.414)
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        self.drug_k = torch.empty([175, 8])
        self.mic_k = torch.empty([95, 8])
        for i in range(0, 270):
            if i < 175:
                self.drug_k[i] = x[i]
            else:
                self.mic_k[i - 175] = x[i]
        score = torch.mm(self.drug_k, self.alpha1)
        score = torch.mm(score, self.mic_k.T)

        return score

