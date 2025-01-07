import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from multiskeleton.module_ms_relationdegree import *
from multiskeleton.module_ms import *
from graphrepre.net_iat import *




visimg=0
visimg2=0
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


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


class SingleStageModel(nn.Module):
    '''
    self.labelmodeling = SingleStageModel(A, 5, 128, num_class, num_class)
    '''
    def __init__(self, A, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1) #12 to 128   #[1,12,64] to [1,120,64]
        # print("1----weight", self.conv_1x1.weight.shape)
        # print(self.conv_1x1.weight)
        # print(self.conv_1x1.bias)


        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)]) #numlayers=10
        self.labelmodelgraph = Graphtransformerlabelmodel(A, 64, num_f_maps, num_f_maps)   #A=0 bs =128-64  bs=256-128
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, x_res):
        out = self.conv_1x1(x) #(1,120, 64) #(1, dim, bs)
        # out = out + x_res
        # print("1---------")
        # print(self.conv_1x1.weight)
        # print(out[0])

        for layer in self.layers:
            out = layer(out)    #1, 120, 64]
            out = out.permute(0,2,1)
            out = self.labelmodelgraph(out).permute(0,2,1)   #1, 64, 120  to 1,128,64
            # print("out1------", out1.shape)


        pred = self.conv_out(out) #(1,12, 64)

        return pred


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()


    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        # nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.dropout(out) #(1,64,964)
        return (x + out)



class unit_tcn(nn.Module):
    '''
    Input (N*M，C，T，V)

    '''
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    '''

    input (N*M, C,T,V)
    '''
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding   #16
        self.inter_c = inter_channels

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)


        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))       #change C dim

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
        A = self.A.cuda(x.get_device())   #-------------------cuda
        # print("A----", self.A[0, 0, :])
        A = A + self.PA

        # print("PA----", self.PA[0,0,:])

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]                                          #A #[N,V,V]
            A2 = x.view(N, C * T, V)                                #X [V, C*T, V]
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))  #{N,C, T,V}   W =conv2d(1)   WXA
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)

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


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=8, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A, A_part,  A_joint2part, A_keleton2skeleton_s1, A_keleton2skeleton_s2 = self.graph.A  #(3,18,18)

        self.A_joint2part = Variable(torch.from_numpy(A_joint2part.astype(np.float32)), requires_grad=False)
        self.A_keleton2skeleton_s1 = Variable(torch.from_numpy(A_keleton2skeleton_s1.astype(np.float32)), requires_grad=False)
        self.A_keleton2skeleton_s2 = Variable(torch.from_numpy(A_keleton2skeleton_s2.astype(np.float32)), requires_grad=False)
        # self.A_joint2part = torch.tensor(self.A_joint2part, dtype=torch.float32)
        print("A-----", A.shape)   #[3,8,8]
        print("A_part-----", A_part.shape)  #[3,3,3]
        print("A_joint2part-----", A_joint2part.shape) #[3,8]
        print("A_keleton2skeleton_s1-----", A_keleton2skeleton_s1.shape)  # [8,8]
        print("A_keleton2skeleton_s2-----", A_keleton2skeleton_s2.shape)  # [3,3]


        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)


        '''
        multi-skeleton
        '''
        self.s2_init = AveargeJoint_mouse()

        time1 = 20

        self.j2p_1 = S1_to_S2(t=time1, n_j1=128, n_j2=(640, 128), n_p1=128, n_p2=(640, 128), t_kernel=3,
                              t_stride=(1, 2), t_padding=1, weight=(256, 256))

        self.p2j_1 = S2_to_S1(t=time1, n_p1=128, n_p2=(640, 128), n_j1=128, n_j2=(640, 128), t_kernel=3,
                              t_stride=(1, 2), t_padding=1, weight=(256, 256))

        num_point_s2 = 3
        Time =20
        self.graphlevel_s1 = GraphTransformerEncode(config, A, num_point, Time)
        self.graphlevel_s2 = GraphTransformerEncode(config, A_part, num_point_s2, Time)


        #decode
        self.graphleveldecode_s1 = GraphTransformerDecode(A, num_point, 64*40,  64*40)
        self.graphleveldecode_s2 = GraphTransformerDecode_s2(A, num_point_s2, 64 * 40,  64*40)

        numpool = 6
        # self.updategraph = GraphtransformerUpdategraph(A, numpool, 64*40, 120)
        # self.fc_updategraph = nn.Linear(120, 64*40)
        self.updategraph = GraphTransformerEncode(config, A, numpool, Time)
        self.fc_calcweight = nn.Linear( 64 * 40, 1)


        self.fc_calcweight_s2 = nn.Linear(64 * 40, 1)
        self.fc_self = nn.Linear(64 * 40, 1)


        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)

        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)


        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)


        self.s2_l1 = TCN_GCN_unit(3, 64, A_part, residual=False)
        self.s2_l5 = TCN_GCN_unit(64, 128, A_part, stride=2)
        self.s2_l8 = TCN_GCN_unit(128, 256, A_part, stride=2)

        self.map2joint = Partmap2joint()   # convert 3 to 8
        self.fc = nn.Linear(384, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)



        #------------------------------l2-------------------------------------------

        time2 = 10
        self.j2p_2 = S1_to_S2(t=time2, n_j1=256, n_j2=(640, 128), n_p1=256, n_p2=(640, 128), t_kernel=3,
                              t_stride=(1, 2), t_padding=1, weight=(256, 256))

        self.p2j_2 = S2_to_S1(t=time2, n_p1=256, n_p2=(640, 128), n_j1=256, n_j2=(640, 128), t_kernel=3,
                              t_stride=(1, 2), t_padding=1, weight=(256, 256))
        Time = 10
        # num_point_s2 = 3
        self.graphlevel_s1_l2 = GraphTransformerEncode(config, A, num_point, Time)
        self.graphlevel_s2_l2 = GraphTransformerEncode(config, A_part, num_point_s2, Time)
        #
        numpool = 6
        self.updategraph_l2 = GraphTransformerEncode(config, A, numpool, Time)


        self.labelmodeling = SingleStageModel(A, 5, 128, num_class, num_class)

    def fuse_operation(self, x1, x2,x3, w1,w2):
        x = x1 + w1 * x2+ w2 * x3
        return x

    def forward(self, x, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed, relrec_speed2, relsend_speed2):
        N, C, T, V, M = x.size()  # 64, 3, 300, 18, 2
        self.A_joint2part = self.A_joint2part.cuda(x.get_device())  # -------------------cuda
        self.A_keleton2skeleton_s1 = self.A_keleton2skeleton_s1.cuda(x.get_device())
        self.A_keleton2skeleton_s2 = self.A_keleton2skeleton_s2.cuda(x.get_device())


        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)       # remove?
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)   #(64*2, 3,150,18)--batch64 0n gpu 0.  if 1 gpu, (128*2, 3,40,8)

        x_s2 = self.s2_init(x)    # [128, 3, 40, 3]


        x_s2_ori = self.map2joint(x_s2)
        x_s1_ori = x.permute(0, 3, 1, 2).contiguous().view(N * M, V, C * T)  # [128, 8, 3*40]
        x_s2_ori = x_s2_ori.permute( 0, 3, 1, 2).contiguous().view(N * M, V, C * T)  # [128, 8, 3*40]
        similarity_ori = torch.cosine_similarity(x_s1_ori, x_s2_ori, dim=2)  #[128,8]


        x_s1_1 = self.l1(x)      #[128, 64, 40, 8]
        x_s2_1 = self.s2_l1(x_s2)  #[128, 64, 40, 3]

        x_s1_1 = self.l5(x_s1_1)  # [128, 128, 20, 8]
        x_s2_1 = self.s2_l5(x_s2_1)  # [128, 128, 20, 3

       # multi-skeleton
        c12_1, x_s2_t1t2_afterinter = self.j2p_1(x_s1_1, x_s2_1, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed, self.A_joint2part, self.A_keleton2skeleton_s1, self.A_keleton2skeleton_s2)  # s1 to s2 [128, 64, 40, 3]
        r12_1, x_s1_t1t2_afterinter = self.p2j_1(x_s2_1, x_s1_1, relrec_s2, relsend_s2, relrec_s1, relsend_s1, relrec_speed, relsend_speed, self.A_joint2part, self.A_keleton2skeleton_s1, self.A_keleton2skeleton_s2)  # s2 to s1 [128, 64, 40 ,8]
        x_s1_1 = self.fuse_operation(x_s1_1, x_s1_t1t2_afterinter, r12_1, 1,1)  # cross_w 0.3
        x_s2_1 = self.fuse_operation(x_s2_1, x_s2_t1t2_afterinter, c12_1, 1,1)



        #  # self-supervisie
        x_s2_self = self.map2joint(x_s2_1)
        x_s2_self = x_s2_self.permute(0, 3, 1, 2).contiguous().view(N * M, V,128 * 20)  # [128, 8, 64*40]
        x_s1_self = x_s1_1
        x_s1_self = x_s1_self.permute(0, 3, 1, 2).contiguous().view(N * M, V, 128 * 20)  # [128, 8, 64*40]
        x_s12_self = x_s1_self - x_s2_self
        similarity_l1 = self.fc_self(x_s12_self).contiguous().view(N * M, V)  # (N, 8,1) to (N, 8)
        # similarity_l1 = torch.softmax(similarity_l1,1)




        for j in range(1):
            # graph representation

            C_graphlevel, T_graphlevel = x_s1_1.shape[1:3]
            x_s1_1_graphlevel = x_s1_1.permute(0,3, 1,2).contiguous().view(N * M, V, C_graphlevel*T_graphlevel)     #[128, 8, 64*40]
            x_s1_1_graphlevel_ori = x_s1_1_graphlevel
            x_s1_1_graphlevel = self.graphlevel_s1(x_s1_1_graphlevel,relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed) #[128,1, 2560]


            C_graphlevel, T_graphlevel = x_s2_1.shape[1:3]
            x_s2_1_graphlevel = x_s2_1.permute(0, 3, 1, 2).contiguous().view(N * M, 3, C_graphlevel * T_graphlevel)  # [128, 3, 64*40]
            x_s2_1_graphlevel_ori = x_s2_1_graphlevel
            x_s2_1_graphlevel = self.graphlevel_s2(x_s2_1_graphlevel,  relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed)   #[128,1, 2560]

            #update graph de and node
            #--------s1-------------------
            x_s1_1_ave = x_s1_1_graphlevel_ori.mean(1)   #(128,2560)
            x_s1_1_ave = x_s1_1_ave.unsqueeze(1)  # [128, 1,2560]
            x_s1_1_max, index = torch.max(x_s1_1_graphlevel_ori, 1)
            x_s1_1_max = x_s1_1_max.unsqueeze(1)
            x_s1_1_ave_updategraph = torch.cat((x_s1_1_graphlevel, x_s1_1_ave, x_s1_1_max), 1)   #3

            #--------s2-------------------
            x_s2_1_ave = x_s2_1_graphlevel_ori.mean(1)  # (128,2560)
            x_s2_1_ave = x_s2_1_ave.unsqueeze(1)  # [128, 1,2560]
            x_s2_1_max, index = torch.max(x_s2_1_graphlevel_ori, 1)
            x_s2_1_max = x_s2_1_max.unsqueeze(1)

            x_s2_1_ave_updategraph = torch.cat((x_s2_1_graphlevel, x_s2_1_ave, x_s2_1_max),1)  # [128, 3,2560]--------self

            #cross skeleton update
            x_s1_1_ave_updategraph = torch.cat((x_s1_1_ave_updategraph, x_s2_1_ave_updategraph), 1)  #[128, 6,2560]

            #-------update s1 graph
            x_s1_1_ave_updategraph = self.updategraph(x_s1_1_ave_updategraph, relrec_s1, relsend_s1, relrec_s2, relsend_s2,relrec_speed, relsend_speed)  # [128,1, 2560]

            x_s1_1_ave_updategraph_out = x_s1_1_ave_updategraph.permute(0, 2, 1).contiguous().view(N * M, C_graphlevel, T_graphlevel,1)  # (128,128,20,1)
            c_new = x_s1_1_ave_updategraph_out.size(1)
            x_s1_1_ave_updategraph_out = x_s1_1_ave_updategraph_out.view(N, M, c_new, -1)  # (64, 2, 128, 20)
            x_s1_1_ave_updategraph_out = x_s1_1_ave_updategraph_out.mean(3).mean(1)  # (64, 128)
            x_res = x_s1_1_ave_updategraph_out.unsqueeze(0).permute(0, 2, 1)


            x_s1_1_ave_updategraph_repeat = x_s1_1_ave_updategraph.repeat(1, 7, 1)

            # -------update s2 graph
            x_s2_1_ave_updategraph_repeat = x_s1_1_ave_updategraph.repeat(1, 3, 1)

            # decode s1
            x_s1_1_ave_updategraphdecode = self.graphleveldecode_s1(x_s1_1_ave_updategraph, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed)  # [128,8, 2560]  size--30M
            # print("decode-------s1", x_s1_1_ave_updategraphdecode.shape)

            # decode s2
            x_s2_1_ave_updategraphdecode = self.graphleveldecode_s2(x_s1_1_ave_updategraph, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed)  # [128,3, 2560]  size--30M
            # print("decode-------s2", x_s1_1_ave_updategraphdecode_s2.shape)


            ###update s1 node
            weight_updatenode = torch.sigmoid(self.fc_calcweight(x_s1_1_ave_updategraph_repeat))  # noQTCNdecoder2
            x_s1_1_ave_updatenode = weight_updatenode * x_s1_1_graphlevel_ori + x_s1_1_graphlevel_ori + x_s1_1_ave_updategraphdecode  # (128,8,2560)
            x_s1_1 =   x_s1_1_ave_updatenode.permute(0, 2, 1).contiguous().view(N * M, C_graphlevel, T_graphlevel, V) #[128, 128, 20 ,8]

            # global visimg
            # F = FeatureVisualization(x_s1_1, visimg)
            # F.save_feature_to_img()
            # visimg = visimg + 1


            ###update s2 node
            weight_updatenode = torch.sigmoid(self.fc_calcweight_s2(x_s2_1_ave_updategraph_repeat))  # (128, 3 ,1)
            x_s2_1_ave_updatenode = weight_updatenode  * x_s2_1_graphlevel_ori + x_s2_1_graphlevel_ori +  (1 - weight_updatenode) * x_s2_1_ave_updategraphdecode  # (128,3,2560)
            x_s2_1 = x_s2_1_ave_updatenode.permute(0, 2, 1).contiguous().view(N * M, C_graphlevel, T_graphlevel, 3)  ##[128, 128, 20 ,3]


        #------------l2,no decoder----------------------

        x_s1_2 = self.l8(x_s1_1)    #[128,128,20,8]
        x_s2_2 = self.s2_l8(x_s2_1)

        c12_1, x_s2_t1t2_afterinter = self.j2p_2(x_s1_2, x_s2_2, relrec_s1, relsend_s1, relrec_s2, relsend_s2,relrec_speed2, relsend_speed2,  self.A_joint2part, self.A_keleton2skeleton_s1, self.A_keleton2skeleton_s2)  # s1 to s2 [128, 64, 40, 3]
        r12_1, x_s1_t1t2_afterinter = self.p2j_2(x_s2_2, x_s1_2, relrec_s2, relsend_s2, relrec_s1, relsend_s1, relrec_speed2, relsend_speed2, self.A_joint2part, self.A_keleton2skeleton_s1, self.A_keleton2skeleton_s2)  # s2 to s1 [128, 64, 40 ,8]
        x_s1_2 = self.fuse_operation(x_s1_2, x_s1_t1t2_afterinter, r12_1, 1,1)  # cross_w 0.3
        x_s2_2 = self.fuse_operation(x_s2_2, x_s2_t1t2_afterinter, c12_1, 1, 1)

        C_graphlevel, T_graphlevel = x_s1_2.shape[1:3]
        x_s1_2_graphlevel = x_s1_2.permute(0, 3, 1, 2).contiguous().view(N * M, V, C_graphlevel * T_graphlevel)  # [128, 8, 128*20]
        x_s1_2_graphlevel_ori = x_s1_2_graphlevel
        x_s1_2_graphlevel = self.graphlevel_s1_l2(x_s1_2_graphlevel, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed)  # [128,1, 2560]

        C_graphlevel, T_graphlevel = x_s2_2.shape[1:3]
        x_s2_2_graphlevel = x_s2_2.permute(0, 3, 1, 2).contiguous().view(N * M, 3, C_graphlevel * T_graphlevel)  # [128, 3, 64*40]
        x_s2_2_graphlevel_ori = x_s2_2_graphlevel
        x_s2_2_graphlevel = self.graphlevel_s2_l2(x_s2_2_graphlevel, relrec_s1, relsend_s1, relrec_s2, relsend_s2,  relrec_speed, relsend_speed)  # [128,1, 2560]



        # update graph de and node
        # ------------s1--------------
        x_s1_2_ave = x_s1_2_graphlevel_ori.mean(1)  # (128,2560)
        x_s1_2_ave = x_s1_2_ave.unsqueeze(1)  # [128, 1,2560]
        x_s1_2_max, index = torch.max(x_s1_2_graphlevel_ori, 1)
        x_s1_2_max = x_s1_2_max.unsqueeze(1)
        x_s1_2_ave_updategraph = torch.cat((x_s1_2_graphlevel, x_s1_2_ave, x_s1_2_max), 1)

        # ------------s2-------------------
        x_s2_2_ave = x_s2_2_graphlevel_ori.mean(1)  # (128,2560)
        x_s2_2_ave = x_s2_2_ave.unsqueeze(1)  # [128, 1,2560]
        x_s2_2_max, index = torch.max(x_s2_2_graphlevel_ori, 1)
        x_s2_2_max = x_s2_2_max.unsqueeze(1)
        x_s2_2_ave_updategraph = torch.cat((x_s2_2_graphlevel, x_s2_2_ave, x_s2_2_max), 1)


        # cross skeleton update
        x_s1_2_ave_updategraph = torch.cat((x_s1_2_ave_updategraph, x_s2_2_ave_updategraph), 1)  # [128, 6,2560]

        # -------update s1 graph
        x_s1_2_ave_updategraph = self.updategraph_l2(x_s1_2_ave_updategraph, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed)  # (128, 1, 2560)


        x_s1_2_ave_updategraph = x_s1_2_ave_updategraph.permute(0,2,1).contiguous().view(N * M, C_graphlevel, T_graphlevel, 1)  #(128,128,20,1)
        c_new = x_s1_2_ave_updategraph.size(1)
        x_s1_2_ave_updategraph = x_s1_2_ave_updategraph.view(N, M, c_new, -1)  # (64, 2, 128, 20)
        # x_s1_2_ave_updategraph = x_s1_2_ave_updategraph.view(N, M * c_new, -1)

        x_s1_2_ave_updategraph = x_s1_2_ave_updategraph.mean(3).mean(1) # (64, 128)
        x_s1_2_ave_updategraph = torch.cat((x_s1_2_ave_updategraph, x_s1_1_ave_updategraph_out), 1)  # (64, 384)


        x_res = x_s1_2_ave_updategraph.unsqueeze(0).permute(0, 2, 1)  # （1，128，64）
        x = self.fc(x_s1_2_ave_updategraph)  # (64, 12)
        # x = torch.softmax(x, -1)
        x_TCN = x

        return x, x_TCN, similarity_ori, similarity_l1


