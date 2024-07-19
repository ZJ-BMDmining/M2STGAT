import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool as gap
from torch.nn import LayerNorm, Parameter
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict

from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
import unittest
import torch
from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm



def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class Fusion(nn.Module):
    def __init__(self, num_class, num_views, hidden_dim, dropout, in_dim):
        super().__init__()
        self.gat1 = GAT(dropout=0.2, alpha=0.6, dim=1000)
        self.TCN = TCN(input_size=1000, output_size=1000, num_channels=[64], kernel_size=3, dropout=0.1)

        # self.rnn =nn.RNN(input_size=hidden_dim[0], hidden_size=hidden_dim[1], num_layers=1, batch_first=True)
        # self.rnn = nn.RNN(input_size=64, hidden_size=64, num_layers=1, batch_first=True)

        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1 , batch_first=True)

        self.mcb = CompactBilinearPooling(1000, 64, 1064).cuda()

        # self.fc_gru = nn.Linear(self.hidden_dim, num_class)

        # self.rnn = nn.RNN(input_size=your_input_size, hidden_size=rnn_hidden_dim, batch_first=True)
        # self.fc_rnn = nn.Linear(rnn_hidden_dim, num_class)

        # self.views = len(in_dim)

        self.classes = num_class
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.FeatureInforEncoder1 = nn.ModuleList(
            [LinearLayer(in_dim[0], in_dim[0]) ])
        self.TCPConfidenceLayer1 = nn.ModuleList([LinearLayer(hidden_dim[0], 1) ])

        self.FeatureInforEncoder2 = nn.ModuleList(
            [LinearLayer(in_dim[1], in_dim[1])])
        self.TCPConfidenceLayer2 = nn.ModuleList([LinearLayer(hidden_dim[1], 1)])

        self.TCPClassifierLayer1 = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) ])
        self.TCPClassifierLayer2 = nn.ModuleList([LinearLayer(hidden_dim[1], num_class) ])


        self.MMClasifier1 = []
        self.MMClasifier1.append(LinearLayer(hidden_dim[0], num_class))
        self.MMClasifier1 = nn.Sequential(*self.MMClasifier1)


        self.MMClasifier2 = []
        self.MMClasifier2.append(LinearLayer(hidden_dim[1], num_class))
        self.MMClasifier2 = nn.Sequential(*self.MMClasifier2)

    def forward(self ,omic1, adj1,tcn_data=None, label=None, tcn_infer=False,infer=False):

        if tcn_infer:
            tcn_fe = self.TCN(tcn_data.transpose(0, 1))
            omic1=tcn_fe
            omic1=omic1.unsqueeze(-1)
            tcn_out=omic1

            output1, gat_output1 = self.gat1(tcn_out, adj1)
            tcn_out=tcn_out.squeeze(-1)

        else:
            output1, gat_output1 = self.gat1(omic1, adj1)

        rnn_output, rnn_hidden = self.gru(output1)

        if tcn_infer:
            tcn_out = tcn_out.squeeze(-1)

            concatenated_tensor = torch.cat((tcn_out, rnn_output), dim=1)
            # self.mcb = CompactBilinearPooling(1000, 64, 1064).cuda()
            # concatenated_tensor = self.mcb(tcn_out,rnn_output)

            feature = dict()
            feature[0]=concatenated_tensor

            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_function = nn.CrossEntropyLoss()
            #
            FeatureInfo, TCPLogit, TCPConfidence = dict(), dict(), dict()
            feature[0] = F.relu(feature[0])
            feature[0] = F.dropout(feature[0], self.dropout, training=self.training)
            # TCPLogit[0] = self.TCPClassifierLayer2[0](feature[0])
            # TCPConfidence[0] = self.TCPConfidenceLayer2[0](feature[0])
            # feature[0] = feature[0] * TCPConfidence[0]
            MMfeature=feature[0]
            MMlogit = self.MMClasifier2(MMfeature)


        else:
            feature = dict()

            feature[0] = rnn_output

            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss_function = nn.CrossEntropyLoss()
            #
            FeatureInfo, TCPLogit, TCPConfidence = dict(), dict(), dict()
            feature[0] = F.relu(feature[0])
            feature[0] = F.dropout(feature[0], self.dropout, training=self.training)
            # TCPLogit[0] = self.TCPClassifierLayer1[0](feature[0])
            # TCPConfidence[0] = self.TCPConfidenceLayer1[0](feature[0])
            # feature[0] = feature[0] * TCPConfidence[0]
            MMfeature = feature[0]
            MMlogit = self.MMClasifier1(MMfeature)



        if infer:
            return MMlogit



        MMLoss = torch.mean(criterion(MMlogit, label))
        loss_gat = loss_function(output1,label)
        gat_loss = dict()
        gat_loss[0] = loss_gat
        # for view in range(self.views):
        #     MMLoss = MMLoss + gat_loss[view]
        #     pred = F.softmax(TCPLogit[view], dim=1)
        #     p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
        #     confidence_loss = torch.mean(
        #         F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
        #     MMLoss = MMLoss + confidence_loss
        # return MMLoss, MMlogit, gat_output1, output1

        MMLoss = MMLoss + gat_loss[0]
        # pred = F.softmax(TCPLogit[0], dim=1)
        # p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
        # confidence_loss = torch.mean(
        #          F.mse_loss(TCPConfidence[0].view(-1), p_target) + criterion(TCPLogit[0], label))
        # MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit, gat_output1, output1

    def infer(self, omic1, adj1,tcn_data,tcn_infer=False):    #omic1, adj1,tcn_data=None, label=None, infer=False,tcn_infer=False
        MMlogit = self.forward(omic1, adj1, tcn_data, tcn_infer=tcn_infer,infer=True)
        return MMlogit


class GAT(nn.Module):
    def __init__(self, dropout, alpha, dim):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.act = define_act_layer(act_type='none')
        self.dim = dim
        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [1000, 128, 64, 32]

        self.attentions1 = [GraphAttentionLayer(
            1, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)

        self.attentions2 = [GraphAttentionLayer(
            self.nhids[0] * self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)

        self.attentions3 = [GraphAttentionLayer(
            self.nhids[1] * self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[2])]
        for i, attention3 in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention3)

        self.dropout_layer = nn.Dropout(p=self.dropout)


        self.pool1 = torch.nn.Linear(self.nhids[0] * self.nheads[0], 1)
        self.pool2 = torch.nn.Linear(self.nhids[1] * self.nheads[1], 1)
        self.pool3 = torch.nn.Linear(self.nhids[2] * self.nheads[2], 1)

        lin_input_dim = 3 * self.dim
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc1.apply(xavier_init)

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], 6))
        self.fc5.apply(xavier_init)

    def forward(self, x, adj):


        x0 = torch.mean(x, dim=-1)
        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)

        

        x1 = self.pool1(x).squeeze(-1)
        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)

        x2 = self.pool2(x).squeeze(-1)
        x = torch.cat([x0, x1, x2], dim=1)
        print("x:",x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc3(x)
        x = self.fc4(x1)

        output = x1
        gat_output = x

        return output, gat_output


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input, adj):
        """
        input: mini-batch input. size: [batch_size, num_nodes, node_feature_dim]
        adj:   adjacency matrix. size: [num_nodes, num_nodes].  need to be expanded to batch_adj later.
        """
        h = torch.matmul(input, self.W)
        bs, N, _ = h.size()

        a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1,
                                                                                                     2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        batch_adj = torch.unsqueeze(adj, 0).repeat(bs, 1, 1)


        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(batch_adj > 0, e, zero_vec)
        attention = self.dropout_layer(F.softmax(attention, dim=-1))  # [bs, N, N]
        h_prime = torch.bmm(attention, h)  # [bs, N, F]



        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred
