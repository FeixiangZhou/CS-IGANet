import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn import GCNConv, GINConv
# from torch_geometric.utils import degree
# from torch_geometric.utils import to_dense_batch
# from model.agcn_multiskeleton import unit_gcn
from ogb.graphproppred.mol_encoder import BondEncoder
from torch.autograd import Variable
import numpy as np
from multiskeleton.module_ms import *

class MAB(nn.Module):
    '''
    self.mab = MAB(A, dim, dim, _output_dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv, _model_str = 'GMPool_I')
    update / inter互动 【TCN（intra），inter】后再TCN降维度
    '''
    def __init__(self, A, _num_point, num_seeds,  dim_Q, dim_K, dim_V, num_heads, ln=False, cluster=False, conv=None, _model_str = None, _T =40):  #384
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self._model_str = _model_str
        self.T = _T
        self.num_seeds = num_seeds

        self.fc_k, self.fc_v = self.get_fc_kv(A, _T, _num_point, dim_K, dim_V, conv, _model_str)


        if(_num_point ==7 and  _T==20):
            if (self._model_str == 'GMPool_G' or self._model_str == 'GMPool_I'):
                self.trans_inter_k, self.trans_inter_v = self.trans_inter_kv(A, _T, _num_point, dim_K, dim_V, conv, _model_str)
                self.interske_k, self.interske_v = self.get_interskeleton_kv(A, _T, _num_point, dim_K, dim_V, conv, _model_str)



        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = self.get_fc_o(A, _T, num_seeds, dim_K, dim_V, conv, _model_str)  # num_seeds = after downscale


        self.softmax_dim = 2
        if cluster == True:
            self.softmax_dim = 1

    def forward(self, Q, K, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask=None, graph=None, return_attn=False):
        '''
         return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attn)
        :param Q: #[1,72,384]
        :param K: #[1,316,384]
        :param attention_mask: 
        :param graph: 
        :param return_attn: 
        :return:
        GM_G-- Q (N, 2, 2560) K (N, 8, 2560)  V(N, 8, 2560)  Out: (N, 2, 2560)

        self -- Q（N， 2 2560）to （N， 2， 128）   K (N， 2，2560 ) to  (N， 2，128 )  V (N， 2，2560 ) to  (N， 2，128 )   Out: (N, 2, 128)

        GM—I -- Q （N， 1 2560）   K (N， 2，128 ) to  (N， 2，2560 )  V (N， 2，128 ) to  (N， 2，2560 )   Out: (N, 1, 2560)
        '''

        # Q = self.fc_q(Q)
        # print("GCN-------Q", Q.shape)
        # Adj: Exist (graph is not None), or Identity (else)
        if graph is not None:

            (x, edge_index, batch) = graph  #[316, 384], [2,1496] [316]
            K, V = self.fc_k(x, edge_index), self.fc_v(x, edge_index)#k [1,316,384]  v [1,316,384]
            K, _ = to_dense_batch(K, batch)  #[1,316,384]
            V, _ = to_dense_batch(V, batch)

        else:
            #----GCN-----
            # print("input---------K", K.shape)
            N =  K.shape[1]   #Node number
            T =self.T
            C = int(K.shape[-1] / T)

            if( self._model_str == None):
                K, V = self.fc_k(K), self.fc_v(K)
                # print("K1----------", K.shape)
            else:
                K = K.permute(0, 2, 1).contiguous().view(K.shape[0], C, T, N)
                K_ori = K

                K, V = self.fc_k(K), self.fc_v(K)   #(N,C,T,V)
                K_intra = K
                V_intra = V

                K = K.permute(0, 3, 1, 2).contiguous().view(K.shape[0], N, -1)  #(N, 8, 120)
                V = V.permute(0, 3, 1, 2).contiguous().view(V.shape[0], N, -1)
                if( N == 7 and T ==20):
                    K_interskeleton, V_interskeleton = self.interske_k(K_ori, relrec_s1, relsend_s1, relrec_speed, relsend_speed), self.interske_v(K_ori, relrec_s1, relsend_s1, relrec_speed, relsend_speed )  # (N,C,T,V)

                   #sum
                    # K = K_intra + K_interskeleton
                    # V = V_intra + V_interskeleton

                   #cat
                    K = torch.cat((K_intra, K_interskeleton), 1)  # (N,128*2,20, 7)
                    V = torch.cat((V_intra, V_interskeleton), 1)  # (N,128*2,20, 7)
                    K, V = self.trans_inter_k(K), self.trans_inter_v(V)  #(N,128,20, 7)

                    K = K.permute(0, 3, 1, 2).contiguous().view(K.shape[0], N, -1)  # (N,7, 128*20)
                    V = V.permute(0, 3, 1, 2).contiguous().view(V.shape[0], N, -1)
                    # print("K_interskeleton_N8----------", K_interskeleton.shape)

                # if N == 3:
                #     K_interskeleton, V_interskeleton = self.interske_k(K_ori, K, relrec_s1, relsend_s1, relrec_speed, relsend_speed), self.interske_v(K_ori, K, relrec_s1, relsend_s1, relrec_speed, relsend_speed )  # (N,C,T,V)
                #     K = K + K_interskeleton
                #     V = V + V_interskeleton
                    # print("K_interskeleton_N3----------", K_interskeleton.shape)
            # print("GCN-------K2", K.shape)

        dim_split = self.dim_V // self.num_heads   #384/1
        Q_ = torch.cat(Q.split(dim_split, 2), 0) #【1，72，384】
        K_ = torch.cat(K.split(dim_split, 2), 0)  #【1，316，384】
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, self.softmax_dim)  #【1，72，316】
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), self.softmax_dim)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)   #【1，72，384】
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)


        T = self.T
        C = int(K.shape[-1] / T)
        N =   self.num_seeds
        O_gcn= O.permute(0, 2, 1).contiguous().view(K.shape[0], C, T, N)
        O_gcn = self.fc_o(O_gcn)  #(N,C,T,V)
        O_gcn = O_gcn.permute(0, 3, 1, 2).contiguous().view(K.shape[0], N, -1)  # (N, 8, 64*40)
        O = O + F.relu(O_gcn)

        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if return_attn:
            return O, A
        else:
            return O

    def get_fc_kv(self, A, _T, _num_point, dim_K, dim_V, conv, _model_str):
        # print("get_fc_kv1-------_model_str", _model_str)
        # print("get_fc_kv1-------conv", conv)

        if conv == 'GCN':
            dim, dim2 = 0, 0
            if _T == 40:
                dim =64
                dim2 = 64
            if _T == 20:
                dim = 128
                dim2 = 128
            if _T == 10:
                dim = 256
                dim2 = 256


            if(_model_str== 'GMPool_G' or _model_str== 'GMPool_U'):
                # print("GMPool_G1-------", _model_str)
                fc_k =  unit_gcn(dim, dim2, A, _num_point, _model_str)   # dim=64
                fc_v =  unit_gcn(dim, dim2, A, _num_point, _model_str)  # dim=64


            elif(_model_str== 'GMPool_I'):
                # print("GMPool_I1-------", _model_str)
                fc_k =  unit_gcn(dim2, dim, A,  _num_point, _model_str)   # dim=64
                fc_v =  unit_gcn(dim2, dim, A,  _num_point, _model_str)  # dim=64


            elif (_model_str == 'GMPool_GCN'): #decoder
                # print("GMPool_I1-------", _model_str)
                fc_k = unit_gcn(dim2, dim, A, _num_point, _model_str)  # dim=64
                fc_v = unit_gcn(dim2, dim, A, _num_point, _model_str)  # dim=64

                # fc_k = TCN_GCN_unit(dim, dim2, A, _num_point, _model_str)  # dim=64
                # fc_v = TCN_GCN_unit(dim, dim2, A, _num_point, _model_str)  # dim=64

            elif (_model_str == 'GMPool_TCN'): #decoder
                # print("GMPool_I1-------", _model_str)
                fc_k = unit_tcn(dim, dim, stride=1)  # dim=64
                fc_v = unit_tcn(dim, dim, stride=1)  # dim=64
            else:  #_model_str='GMPool_label'
                fc_k = unit_gcn(dim_K, dim_V, A, _num_point, _model_str)  # dim=64   #graph-level inter
                fc_v = unit_gcn(dim_K, dim_V, A, _num_point, _model_str)  # dim=6

        else:

            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)

        return fc_k, fc_v

    def trans_inter_kv(self, A, _T, _num_point, dim_K, dim_V, conv, _model_str):
        # print("get_fc_kv1-------_model_str", _model_str)
        # print("get_fc_kv1-------conv", conv)

        if conv == 'GCN':
            dim, dim2 = 0, 0
            if _T == 40:
                dim = 128
                dim2 = 64
            if _T == 20:
                dim = 256
                dim2 = 128
            if _T == 10:
                dim = 512
                dim2 = 256

            if (_model_str == 'GMPool_G' or _model_str == 'GMPool_U'):
                # print("GMPool_G1-------", _model_str)
                # fc_k = unit_gcn(dim, dim2, A, _num_point, _model_str)  # dim=64  GCN
                # fc_v = unit_gcn(dim, dim2, A, _num_point, _model_str)  # dim=64

                fc_k = unit_tcn(dim, dim2, kernel_size=1, stride=1)  # dim=64   TCN
                fc_v = unit_tcn(dim, dim2, kernel_size=1, stride=1)  # dim=64   TCN



        else:

            fc_k = nn.Linear(dim_K, dim_V)
            fc_v = nn.Linear(dim_K, dim_V)

        return fc_k, fc_v

    def get_fc_o(self, A, _T, num_seeds, dim_K, dim_V, conv, _model_str):
        # print("get_fc_kv1-------_model_str", _model_str)
        # print("get_fc_kv1-------conv", conv)
        if conv == 'GCN':
            dim, dim2 = 0, 0
            if _T == 40:
                dim =64
                dim2 = 64
            if _T == 20:
                dim = 128
                dim2 = 128
            if _T == 10:
                dim = 256
                dim2 = 256

            if(_model_str== 'GMPool_G'):  # out points =3
                # print("GMPool_G1-------", _model_str)
                # fc_o =  unit_gcn(dim, dim2, A, num_seeds, _model_str, flag=False)   # dim=64
                fc_o = unit_tcn(dim, dim, stride=1)  # dim=64


            elif(_model_str== 'GMPool_I'): # out points =1
                # print("GMPool_I1-------", _model_str)
                fc_o = unit_tcn(dim, dim, stride=1)  # dim=64

            elif (_model_str == 'GMPool_GCN'):  #  out points=7
                # print("GMPool_I1-------", _model_str)
                # fc_o = unit_gcn(dim2, dim, A, num_seeds, _model_str)  # dim=64
                fc_o = unit_tcn(dim2, dim, stride=1)  # dim=64

            elif (_model_str == 'GMPool_TCN'): #out points=3
                # print("GMPool_I1-------", _model_str)
                # fc_o = unit_gcn(dim2, dim, A, num_seeds, _model_str)  # dim=64
                fc_o = unit_tcn(dim2, dim, stride=1)  # dim=64


            else:  #_model_str='GMPool_label'
                fc_o = unit_gcn(dim_K, dim_V, A, num_seeds, _model_str)  # dim=64   #graph-level inter

        else:

            fc_o = nn.Linear(dim_K, dim_V)


        return fc_o

    def get_interskeleton_kv(self, A, _T, _num_point, dim_K, dim_V, conv, _model_str):

        if conv == 'GCN':

            dim, dim2 = 0, 0
            if _T == 40:
                dim = 64
                dim2 = 64
            if _T == 20:
                dim = 128
                dim2 = 128
            if _T == 10:
                dim = 256
                dim2 = 256

            if (_model_str == 'GMPool_G'):  # nodes =8 S1 skeleton
                # print("GMPool_G1-------", _model_str)
                # fc_k = unit_gcn(64, 3, A, _num_point, _model_str)  # dim=64
                # fc_v = unit_gcn(64, 3, A, _num_point, _model_str)  # dim=64
                fc_k = interskeleton_interaction(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                 t_stride=(1, 2), t_padding=1, weight=(256, 256))
                fc_v = interskeleton_interaction(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                 t_stride=(1, 2), t_padding=1, weight=(256, 256))

            else:  # nodes =3 S2 skeleton
                # print("GMPool_I1-------", _model_str)
                fc_k = interskeleton_interaction_S2(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                    t_stride=(1, 2), t_padding=1, weight=(256, 256))
                fc_v = interskeleton_interaction_S2(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                    t_stride=(1, 2), t_padding=1, weight=(256, 256))

            return fc_k, fc_v


class PMA(nn.Module):
    '''
    num_point, num_seeds : 8----3   ---> 3-----1
    _num_point, num_seeds :3----2   ---> 2-----1
    '''
    def __init__(self, A,  dim, _output_dim, num_heads, _num_point, num_seeds, ln=False, cluster=False, mab_conv=None, _model_str = 'GMPool_G', _T = 40):# 384, 1, 72 , GCN
        super(PMA, self).__init__()
        # self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))  #[1,72,384]
        # nn.init.xavier_uniform_(self.S)

        if(_model_str == 'GMPool_G'):
            self.S = nn.Parameter(torch.Tensor(_num_point, num_seeds))  # [1,316,72]  S2
            # self.S = nn.Parameter(torch.Tensor(_num_point, num_seeds))

        else:
            # self.S = nn.Parameter(torch.Tensor(3, num_seeds))  # [1,72,1] #8-3-1
            self.S = nn.Parameter(torch.Tensor(_num_point, num_seeds))  # [1,72,1] #8-3-1 #s2
        nn.init.xavier_uniform_(self.S)

        if (_model_str == 'GMPool_G'):
            self.mab = MAB(A, _num_point, num_seeds, dim, dim, _output_dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv, _model_str = 'GMPool_G', _T = _T)
        else:
            self.mab = MAB(A, _num_point, num_seeds, dim, dim, _output_dim, num_heads, ln=ln, cluster=cluster, conv=mab_conv, _model_str = 'GMPool_I', _T = _T)
        
    def forward(self, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask=None, graph=None, return_attn=False):
        '''
           batch_x = self.pools[_index](batch_x, attention_mask=extended_attention_mask, graph=(x, edge_index, batch))
        :param X: [1,316,384]
        :param attention_mask: 【1，1，316】
        :param graph: (x, edge_index, batch)
        :param return_attn:
        :return:
        '''
        # temp = self.S.repeat(X.size(0), 1, 1)    #Q
        # temp = self.S.repeat(X.size(0), 1, 1)
        X_Q = X.permute(0, 2, 1)
        # print("PMA--------", X_Q.shape, self.S.shape)
        X_Q = torch.matmul(X_Q, self.S)
        X_Q = X_Q.permute(0, 2, 1) #[1,3,2560]
        # print("PMA--------", X_Q.shape)


        # return self.mab(self.S.repeat(X.size(0), 1, 1), X, attention_mask, graph, return_attn)
        # print("PMA-------X", X.shape)
        return self.mab(X_Q, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask, graph, return_attn)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_gcn(nn.Module):
    '''

    input (N*M, C,T,V)
    '''
    def __init__(self, in_channels, out_channels, A, _num_point, _model_str, coff_embedding=4, num_subset=3, flag=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding   #16
        # inter_channels = 16
        self.flag = flag
        self.inter_c = inter_channels
        self._model_str =_model_str
        self.numpoint = _num_point

        if(_model_str == 'GMPool_G'):

            if flag == True:

                if (_num_point ==3 or _num_point ==7):
                    self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
                    nn.init.constant_(self.PA, 1e-6)
                else:    # _num_point==6 graph
                    self.PA = nn.Parameter(torch.Tensor(3, _num_point, _num_point))  # 2-2
                    nn.init.constant_(self.PA, 1e-6)
            else:   #fc_o
                self.PA = nn.Parameter(torch.Tensor(3, _num_point, _num_point))  # 2-2
                nn.init.constant_(self.PA, 1e-6)
        else:
            # self.PA = nn.Parameter(torch.Tensor(3, 3, 3))
            self.PA = nn.Parameter(torch.Tensor(3, _num_point, _num_point))   #2-2
            nn.init.constant_(self.PA, 1e-6)

        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules(): #
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        # print(x.size)
        A = self.A.cuda(x.get_device())   #-------------------cuda
        if self.flag ==True:
            if (self._model_str == 'GMPool_G'):
                if(self.numpoint==7 or self.numpoint==3):
                    A = A + self.PA
                else:
                    A = self.PA
            else:
                # print("GMPool_I2-----------", self._model_str)
                A = self.PA
        else:
            A = self.PA

        A = A.cuda(x.get_device())

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            # print("A1----", A1.shape)
            # print("A[i]----", A[i].shape)
            A1 = A1 + A[i]                                          #A #[N,V,V]
            A2 = x.view(N, C * T, V)                                #X [V, C*T, V]
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))  #{N,C, T,V}   W =conv2d(1)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class unit_tcn(nn.Module):
    '''
    Input (N*M，C，T，V)

    '''

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # print("scale_liner", x.shape)
        x = self.bn(self.conv(x))
        x = self.relu(x)   #(N, 128, 20, 1)
        # print(x.shape)
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A,  _num_point, _model_str, stride=1, residual=True, flag=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A,  _num_point, _model_str, flag=flag)

        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # print("TCN_GCN_x----", x.shape) #128, 3, 150, 18 [N, C, T, V]

        # print("TCN_GCN1----", self.gcn1(x).shape)  #][128, 64, 150, 18] [N, C, T, V]
        # print("TCN_GCN2----", self.tcn1(self.gcn1(x)).shape) ##][128, 64, 150, 18] [N, C, T, V]
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)