##
#增加了TCN到interskeleton_interaction/ cat---sum
##
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilis import drawmatrix, drawmatrix2, drawmatrix3, drawmatrix4
from multiskeleton.operation_ms import *



visnum=0
visnum2=0

visnum3=0
visnum4=0


def edge2mat2(link, num_joint, num_part):
    A = np.zeros((num_part, num_joint))
    for i, j in link:
        A[i, j] = 1
    return A


def get_graph(num_joint, num_part, prior_link):
    A = edge2mat2(prior_link, num_joint, num_part)

    return A



edge_skeleton2skeleton_s1 = [(0, 0), (1,1),(2,2) ,(3,3), (4,4), (5,5), (6,6)]   #
edge_skeleton2skeleton_s2 = [(0, 0), (1,1),(2,2)]

global_A_keleton2skeleton_s1 = get_graph(7, 7, edge_skeleton2skeleton_s1)
global_A_keleton2skeleton_s2 = get_graph(3, 3, edge_skeleton2skeleton_s2)

global_A_keleton2skeleton_s1 = torch.from_numpy(global_A_keleton2skeleton_s1.astype(np.float32))
global_A_keleton2skeleton_s2 = torch.from_numpy(global_A_keleton2skeleton_s2.astype(np.float32))

class St_gcn(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, t_kernel_size=1, stride=1,
                 dropout=0.5, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0]-1)//2, 0)

        self.gcn = SpatialConv(in_channels, out_channels, kernel_size[1], t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A_skl):
        res = self.residual(x)
        x = self.gcn(x, A_skl)
        x = self.tcn(x) + res
        return self.relu(x)


class SpatialConv(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, 
                 t_kernel_size=1, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*(k_num),
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A_skl):
        x = self.conv(x)                                                   # [64, 128, 49, 21]
        n, kc, t, v = x.size()                                             # n = 64(batchsize), kc = 128, t = 49, v = 21
        x = x.view(n, self.k_num,  kc//(self.k_num), t, v)             # [64, 4, 32, 49, 21]
        A_all = A_skl
        x = torch.einsum('nkctv, kvw->nctw', (x, A_all))
        return x.contiguous()


class DecodeGcn(nn.Module):
    
    def __init__(self, in_channels, out_channels, k_num,
                 kernel_size=1, stride=1, padding=0,
                 dilation=1, dropout=0.5, bias=True):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels*(k_num), 
                              kernel_size=kernel_size,
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation, 
                              bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_skl):      # x: [64, 256, 21] = N, d, V
        x = self.conv(x)
        x = self.dropout(x)
        n, kc, v = x.size()
        x = x.view(n, (self.k_num), kc//(self.k_num), v)          # [64, 4, 256, 21]
        x = torch.einsum('nkcv,kvw->ncw', (x, A_skl))           # [64, 256, 21]
        return x.contiguous()



class AveargeJoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9,10]
        self.left_leg_up = [0,1]
        self.left_leg_down = [2,3]
        self.right_leg_up = [4,5]
        self.right_leg_down = [6,7]
        self.head = [11,12,13]
        self.left_arm_up = [14,15]
        self.left_arm_down = [16,17,18,19]
        self.right_arm_up = [20,21]
        self.right_arm_down = [22,23,24,25]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 3))                                              # [N, C, T, V=1]
        x_leftlegup = F.avg_pool2d(x[:, :, :, self.left_leg_up], kernel_size=(1, 2))                                # [N, C, T, V=1]
        x_leftlegdown = F.avg_pool2d(x[:, :, :, self.left_leg_down], kernel_size=(1, 2))                     # [N, C, T, V=1]
        x_rightlegup = F.avg_pool2d(x[:, :, :, self.right_leg_up], kernel_size=(1, 2))                        # [N, C, T, V=1]
        x_rightlegdown = F.avg_pool2d(x[:, :, :, self.right_leg_down], kernel_size=(1, 2))                   # [N, C, T, V=1]
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 3))                                              # [N, C, T, V=1]
        x_leftarmup = F.avg_pool2d(x[:, :, :, self.left_arm_up], kernel_size=(1, 2))                            # [N, C, T, V=1]
        x_leftarmdown = F.avg_pool2d(x[:, :, :, self.left_arm_down], kernel_size=(1, 4))                 # [N, C, T, V=1]
        x_rightarmup = F.avg_pool2d(x[:, :, :, self.right_arm_up], kernel_size=(1, 2))                        # [N, C, T, V=1]
        x_rightarmdown = F.avg_pool2d(x[:, :, :, self.right_arm_down], kernel_size=(1, 4))               # [N, C, T, V=1]
        x_part = torch.cat((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup, x_leftarmdown, x_rightarmup, x_rightarmdown), dim=-1)               # [N, C, T, V]), dim=-1)        # [N, C, T, 10]
        return x_part


class AveargeJoint_mouse(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = [0, 1, 2]
        self.body = [3,4,5]
        self.tail = [6]

    def forward(self, x):
        x_head = F.avg_pool2d(x[:, :, :, self.head], kernel_size=(1, 3))  # [N, C, T, V=1]
        x_body = F.avg_pool2d(x[:, :, :, self.body], kernel_size=(1, 3))  # [N, C, T, V=1]
        x_tail = F.avg_pool2d(x[:, :, :, self.tail], kernel_size=(1, 1))  # [N, C, T, V=1]
        x_part = torch.cat((x_head, x_body, x_tail),dim=-1)  # [N, C, T, V]), dim=-1)        # [N, C, T, 10]
        return x_part




class AveargePart(nn.Module):

    def __init__(self):
        super().__init__()
        self.torso = [8,9,10,11,12,13]
        self.left_leg = [0,1,2,3]
        self.right_leg = [4,5,6,7]
        self.left_arm = [14,15,16,17,18,19]
        self.right_arm = [20,21,22,23,24,25]
        
    def forward(self, x):
        x_torso = F.avg_pool2d(x[:, :, :, self.torso], kernel_size=(1, 6))                                              # [N, C, T, V=1]
        x_leftleg = F.avg_pool2d(x[:, :, :, self.left_leg], kernel_size=(1, 4))                                # [N, C, T, V=1]
        x_rightleg = F.avg_pool2d(x[:, :, :, self.right_leg], kernel_size=(1, 4))                        # [N, C, T, V=1]
        x_leftarm = F.avg_pool2d(x[:, :, :, self.left_arm], kernel_size=(1, 6))                            # [N, C, T, V=1]
        x_rightarm = F.avg_pool2d(x[:, :, :, self.right_arm], kernel_size=(1, 6))                        # [N, C, T, V=1]
        x_body = torch.cat((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), dim=-1)               # [N, C, T, V]), dim=-1)        # [N, C, T, 5]
        return x_body



class S1_to_S2(nn.Module):
    '''
            self.j2p_1 = S1_to_S2(n_j1=32, n_j2=(800, 256), n_p1=32, n_p2=(800, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
    '''

    def __init__(self, t, n_j1, n_j2, n_p1, n_p2, t_kernel, t_stride, t_padding, weight):
        super().__init__()
        dim =128
        # self.embed_s1 = S1AttInform(t, n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0.5, nmp=True)
        # self.embed_s2 = S2AttInform(t, n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0.5, nmp=True)
        self.embed_s1 = S1AttInform(t, n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0, nmp=True)
        self.embed_s2 = S2AttInform(t, n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0, nmp=True)
        self.softmax = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.weight2 = nn.Parameter(torch.Tensor(dim, dim))

        self.weight_t12t2 = nn.Parameter(torch.Tensor(7, 256))
        self.weight_t22t1 = nn.Parameter(torch.Tensor(7, 256))

        self.weight_t12t2_s2 = nn.Parameter(torch.Tensor(3, 256))
        self.weight_t22t1_s2 = nn.Parameter(torch.Tensor(3, 256))

        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.weight_t12t2)
        nn.init.xavier_uniform_(self.weight_t22t1)
        nn.init.xavier_uniform_(self.weight_t12t2_s2)
        nn.init.xavier_uniform_(self.weight_t22t1_s2)


        # nn.init.constant_(self.weight, 1e-6)
        # nn.init.constant_(self.weight2, 1e-6)
        # nn.init.constant_(self.weight_t12t2, 1e-6)
        # nn.init.constant_(self.weight_t22t1, 1e-6)
        # nn.init.constant_(self.weight_t12t2_s2, 1e-6)
        # nn.init.constant_(self.weight_t22t1_s2, 1e-6)

        # print(self.weight.shape)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_s1, x_s2, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed, A_joint2part, A_keleton2skeleton_s1, A_keleton2skeleton_s2):                                                           # x_s1: [64, 32, 49, 26]
        # print("S1_to_S2_relrec_s1-----", relrec_s1.shape, relsend_s1.shape, relrec_s2.shape, relsend_s2.shape)                              #[650, 26], [650, 26], [90, 10],[90, 10]
        N, d, T, V = x_s1.size()
        N, d, T, W = x_s2.size()
        # print("x_s1---", x_s1.size())

        x_s1_att = self.embed_s1(x_s1, relrec_s1, relsend_s1, relrec_speed, relsend_speed)                                                              # [64*2, 8, 256] update features
        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2, relrec_speed, relsend_speed)                                                             # (64*2,3, 256)



        '''
        # inter-skeleton ---S1
        '''
        x_s1__att_t1t2 = x_s1_att.contiguous().view(-1, 2, V, 128)
        x_s1_att_t1 = x_s1__att_t1t2[:,0,:,:]
        x_s1_att_t2 = x_s1__att_t1t2[:,1,:,:]
        x_s1_att_t2t1 = torch.cat((x_s1_att_t2, x_s1_att_t1), 0)

        #t1 to t2

        x_s1_att_t1t2_cat = torch.cat((x_s1_att_t1, x_s1_att_t2), 2)  # (N, 7, 256)
        # print(x_s1_att_t1t2_cat[0,0,0:10])
        Att_s1_t12t2 = torch.matmul(self.weight_t12t2, x_s1_att_t1t2_cat.permute(0, 2, 1))  # [7,7]


        Att_s1_t12t2 = self.relu(Att_s1_t12t2)


        global visnum
        # for bnum in range(0,1):
        #     drawmatrix(Att_s1_t12t2_noself[bnum], visnum )
        #     visnum = visnum+1

        Att_s1_ori = A_keleton2skeleton_s1 * 0.5
        Att_s1_t12t2 = self.softmax(Att_s1_t12t2 + Att_s1_ori)
        # print("Att_s1_t12t2-----", Att_s1_t12t2_noself[0,:,:].shape) #(64,7,7)
        # print("Att_s1_t12t2-----", Att_s1_t12t2_noself[0:20,:,:])  # (64,7,7)
        # for bnum in range(0,20):
        #     drawmatrix(Att_s1_t12t2_noself[bnum], bnum)



        N2 = N //2
        x_s1_t1t2 = x_s1.permute(0, 3, 2, 1).contiguous().view(N2, 2,  V, -1)
        x_s1_t1 = x_s1_t1t2[:, 0, :, :]  #[64, 8, 2560]
        x_s1_t2 = x_s1_t1t2[:, 1, :, :]
        x_s1_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t12t2, x_s1_t1))   #t1 to t2  #[64,8,2560]
        x_s1_t2_afterinter = x_s1_t2_interactive + x_s1_t2   # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t2_afterinter------", x_s1_t2_afterinter.shape)

        # t2 to t1
        Att_s1_t22t1 = torch.matmul(self.weight_t22t1, x_s1_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s1_t22t1 = self.relu(Att_s1_t22t1)


        Att_s1_t22t1_noself =  self.softmax(Att_s1_t22t1)
        # print("Att_s1_t22t1_noself-----", Att_s1_t22t1_noself[0, :, :])  # (64,7,7)
        global visnum2
        # for bnum in range(0, 1):
        #     drawmatrix2(Att_s1_t22t1_noself[bnum], visnum2)
        #     visnum2 = visnum2+ 1


        Att_s1_ori = A_keleton2skeleton_s1 * 0.5
        Att_s1_t22t1 = self.softmax(Att_s1_t22t1 + Att_s1_ori)

        N2 = N // 2
        x_s1_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t22t1, x_s1_t2))  # t2 to t1  #[64,8,2560]
        x_s1_t1_afterinter = x_s1_t1_interactive + x_s1_t1  # after interactive   #(64,8,2560) bs =128

        x_s1_t1t2_afterinter = torch.cat((x_s1_t1_afterinter, x_s1_t2_afterinter), 0)  # (128,8,2560) [t1,t2]
        x_s1_t2t1_afterinter = torch.cat((x_s1_t2_afterinter, x_s1_t1_afterinter), 0) #(128,8,2560) [t2,t1]


        x_s1_t1t2_afterinter = x_s1_t1t2_afterinter.permute(0, 2,1).contiguous().view(N,d,T,V)
        x_s1_t2t1_afterinter = x_s1_t2t1_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, V)



        '''
         # inter-skeleton ---S2
        '''
        x_s2__att_t1t2 = x_s2_att.contiguous().view(-1, 2, W, 128)
        x_s2_att_t1 = x_s2__att_t1t2[:, 0, :, :]
        x_s2_att_t2 = x_s2__att_t1t2[:, 1, :, :]

        # t1 to t2
        x_s2_att_t1t2_cat = torch.cat((x_s2_att_t1, x_s2_att_t2), 2)  # (N, 7, 256)
        Att_s2_t12t2 = torch.matmul(self.weight_t12t2_s2, x_s2_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s2_t12t2 = self.relu(Att_s2_t12t2)
        # Att_s2_t12t2_noself = self.softmax(Att_s2_t12t2)
        # print("Att_s2_t12t2_noself-----", Att_s2_t12t2_noself[0, :, :])  # (64,7,7)
        Att_s2_ori = A_keleton2skeleton_s2 * 0.5
        Att_s2_t12t2 = self.softmax(Att_s2_t12t2 + Att_s2_ori)

        N2 = N // 2
        x_s2_t1t2 = x_s2.permute(0, 3, 2, 1).contiguous().view(N2, 2, W, -1)
        x_s2_t1 = x_s2_t1t2[:, 0, :, :]  # [64, 8, 2560]
        x_s2_t2 = x_s2_t1t2[:, 1, :, :]
        x_s2_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s2_t12t2, x_s2_t1))  # t1 to t2  #[64,8,2560]
        x_s2_t2_afterinter = x_s2_t2_interactive + x_s2_t2  # after interactive   #(64,8,2560) bs =128

        # t2 to t1
        Att_s2_t22t1 = torch.matmul(self.weight_t22t1_s2, x_s2_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s2_t22t1 = self.relu(Att_s2_t22t1)
        Att_s2_ori = A_keleton2skeleton_s2 * 0.5
        Att_s2_t22t1 = self.softmax(Att_s2_t22t1 + Att_s2_ori)

        N2 = N // 2
        x_s2_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s2_t22t1, x_s2_t2))  # t1 to t2  #[64,8,2560]
        x_s2_t1_afterinter = x_s2_t1_interactive + x_s2_t1  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t1_afterinter------", x_s1_t1_afterinter.shape)

        x_s2_t1t2_afterinter = torch.cat((x_s2_t1_afterinter, x_s2_t2_afterinter), 0)  # (128,8,2560)
        x_s2_t2t1_afterinter = torch.cat((x_s2_t2_afterinter, x_s2_t1_afterinter), 0)  # (128,8,2560)

        x_s2_t1t2_afterinter = x_s2_t1t2_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, W)
        x_s2_t2t1_afterinter = x_s2_t2t1_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, W)



        # --------------
        '''
        cross-skeleton
        hs2,k *w *hs1,j.T + 0.5* A(k,i)
        '''
        Att = torch.matmul(x_s2_att, self.weight)
        # print("S1_to_S2_weight-----", self.weight)

        Att = torch.matmul(Att, x_s1_att.permute(0,2,1))                                                #[64*2, 3, 8]

        Att_noself = self.softmax(Att)
        # print("Att_s1_t22t1_noself-----", Att_s1_t22t1_noself[0, :, :])  # (64,7,7)
        global visnum3
        # for bnum in range(0, 1):
        #     drawmatrix3(Att_noself[bnum], visnum3)
        #     visnum3 = visnum3 + 1


        # Att_2 = torch.matmul(self.weight2, A_joint2part)
        Att_2 = A_joint2part * 0.5
        Att = self.softmax(Att + Att_2)                                                                 #【128，3，8】
        # print(Att.shape)

        x_s1 = x_s1_t1t2_afterinter.permute(0,3,2,1).contiguous().view(N,V,-1)                                                     # [128, 8, 40, 64] -> [128, 8, 2560]
        x_s2_glb = torch.einsum('nwv, nvd->nwd', (Att, x_s1))                                           # [128, 3, 2560]
        x_s2_glb = x_s2_glb.contiguous().view(N, W, -1, d).permute(0,3,2,1)                             # [128, 3, 2560] -> [128, 3, 40, 64] -> [128, 64, 40, 3], [N,C,T,V]



       #cross S1 t2 to  S2 t1
        Att_s1t2tos2t1 = torch.matmul(x_s2_att, self.weight2)
        # print("S1_to_S2_weight-----", self.weight)
        Att_s1t2tos2t1 = torch.matmul(Att_s1t2tos2t1, x_s1_att_t2t1.permute(0, 2, 1))  # [64*2, 3, 8]
        # print(Att.shape)
        Att_s1t2tos2t1_noself = self.softmax(Att_s1t2tos2t1)

        global visnum4
        # for bnum in range(0, 1):
        #     drawmatrix4(Att_s1t2tos2t1_noself[bnum], visnum4)
        #     visnum4 = visnum4 + 1


        Att_2 = A_joint2part * 0.5
        Att_s1t2tos2t1 = self.softmax(Att_s1t2tos2t1 + Att_2)  # 【128，3，8】
        # print(Att.shape)

        x_s1 = x_s1_t2t1_afterinter.permute(0, 3, 2, 1).contiguous().view(N, V,-1)  # [128, 8, 40, 64] -> [128, 8, 2560]
        x_s2_s1t2tos2t1 = torch.einsum('nwv, nvd->nwd', (Att_s1t2tos2t1, x_s1))  # [128, 3, 2560]
        x_s2_s1t2tos2t1 = x_s2_s1t2tos2t1.contiguous().view(N, W, -1, d).permute(0, 3, 2, 1)
        x_s2_glb = x_s2_glb + x_s2_s1t2tos2t1

        return x_s2_glb, x_s2_t1t2_afterinter


        

class S2_to_S1(nn.Module):
    '''

    self.p2j_1 = S2_to_S1(n_p1=64, n_p2=(12800, 256), n_j1=32, n_j2=(12800, 256), t_kernel=5, t_stride=(1, 2),
                              t_padding=2, weight=(256, 256))
    '''

    def __init__(self, t, n_p1, n_p2, n_j1, n_j2, t_kernel, t_stride, t_padding, weight):
        super().__init__()
        dim = 128
        self.embed_s2 = S2AttInform(t, n_p1, n_p2, t_stride[1], t_kernel, t_padding, drop=0, nmp=True)
        self.embed_s1 = S1AttInform(t, n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0, nmp=True)
        self.softmax = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.weight2 = nn.Parameter(torch.Tensor(dim, dim))
        
        self.weight_t12t2 = nn.Parameter(torch.Tensor(7, 256))
        self.weight_t22t1 = nn.Parameter(torch.Tensor(7, 256))
        self.weight_t12t2_s2 = nn.Parameter(torch.Tensor(3, 256))
        self.weight_t22t1_s2 = nn.Parameter(torch.Tensor(3, 256))
        # self.weight2 = nn.Parameter(torch.Tensor(3, 3))

        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.xavier_uniform_(self.weight_t12t2)
        nn.init.xavier_uniform_(self.weight_t22t1)
        nn.init.xavier_uniform_(self.weight_t12t2_s2)
        nn.init.xavier_uniform_(self.weight_t22t1_s2)

        # nn.init.constant_(self.weight, 1e-6)
        # nn.init.constant_(self.weight2, 1e-6)
        # nn.init.constant_(self.weight_t12t2, 1e-6)
        # nn.init.constant_(self.weight_t22t1, 1e-6)
        # nn.init.constant_(self.weight_t12t2_s2, 1e-6)
        # nn.init.constant_(self.weight_t22t1_s2, 1e-6)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_s2, x_s1, relrec_s2, relsend_s2, relrec_s1, relsend_s1, relrec_speed, relsend_speed, A_joint2part, A_keleton2skeleton_s1,A_keleton2skeleton_s2):

        N, d, T, W = x_s2.size()
        N, d, T, V = x_s1.size()

        x_s2_att = self.embed_s2(x_s2, relrec_s2, relsend_s2, relrec_speed, relsend_speed) ## (128,3, 256)
        x_s1_att = self.embed_s1(x_s1, relrec_s1, relsend_s1, relrec_speed, relsend_speed) # (128,8, 256)
        # Att = self.softmax(torch.matmul(x_s2_att, x_s1_att.permute(0,2,1)).permute(0,2,1))   #(128, 8, 3 )

        '''
         inter-skeleton ---S2
        '''
        x_s2__att_t1t2 = x_s2_att.contiguous().view(-1, 2, W, 128)
        x_s2_att_t1 = x_s2__att_t1t2[:, 0, :, :]
        x_s2_att_t2 = x_s2__att_t1t2[:, 1, :, :]
        x_s2_att_t2t1 = torch.cat((x_s2_att_t2, x_s2_att_t1), 0)

        # t1 to t2
        x_s2_att_t1t2_cat = torch.cat((x_s2_att_t1, x_s2_att_t2), 2)  # (N, 7, 256)
        Att_s2_t12t2 = torch.matmul(self.weight_t12t2_s2, x_s2_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s2_t12t2 = self.relu(Att_s2_t12t2)
        Att_s2_ori = A_keleton2skeleton_s2 * 0.5
        Att_s2_t12t2 = self.softmax(Att_s2_t12t2+Att_s2_ori)



        N2 = N // 2
        x_s2_t1t2 = x_s2.permute(0, 3, 2, 1).contiguous().view(N2, 2, W, -1)
        x_s2_t1 = x_s2_t1t2[:, 0, :, :]  # [64, 8, 2560]
        x_s2_t2 = x_s2_t1t2[:, 1, :, :]
        x_s2_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s2_t12t2, x_s2_t1))  # t1 to t2  #[64,8,2560]
        x_s2_t2_afterinter = x_s2_t2_interactive + x_s2_t2  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t2_afterinter------", x_s1_t2_afterinter.shape)

        # t2 to t1
        Att_s2_t22t1 = torch.matmul(self.weight_t22t1_s2, x_s2_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s2_t22t1 = self.relu(Att_s2_t22t1)
        Att_s2_ori = A_keleton2skeleton_s2 * 0.5
        Att_s2_t22t1 = self.softmax(Att_s2_t22t1 + Att_s2_ori)

        N2 = N // 2
        x_s2_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s2_t22t1, x_s2_t2))  # t1 to t2  #[64,8,2560]
        x_s2_t1_afterinter = x_s2_t1_interactive + x_s2_t1  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t1_afterinter------", x_s1_t1_afterinter.shape)

        x_s2_t1t2_afterinter = torch.cat((x_s2_t1_afterinter, x_s2_t2_afterinter), 0)  # (128,8,2560)
        x_s2_t1t2_afterinter = x_s2_t1t2_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, W)

        x_s2_t2t1_afterinter = torch.cat((x_s2_t2_afterinter, x_s2_t1_afterinter), 0)  # (128,8,2560)
        x_s2_t2t1_afterinter = x_s2_t2t1_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, W)

        # print("x_s2_t1t2_afterinter------", x_s2_t1t2_afterinter.shape)

        '''
               # inter-skeleton ---S1
        '''
        x_s1__att_t1t2 = x_s1_att.contiguous().view(-1, 2, V, 128)
        x_s1_att_t1 = x_s1__att_t1t2[:, 0, :, :]
        x_s1_att_t2 = x_s1__att_t1t2[:, 1, :, :]

        # t1 to t2
        x_s1_att_t1t2_cat = torch.cat((x_s1_att_t1, x_s1_att_t2), 2)  # (N, 7, 256)
        Att_s1_t12t2 = torch.matmul(self.weight_t12t2, x_s1_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s1_t12t2 = self.relu(Att_s1_t12t2)
        # print("s2to21_weight_t12t2----", self.weight_t12t2[:, 0:10])
        # Att_s1_t12t2 = self.softmax(Att_s1_t12t2)
        # print("Att_s1_t12t2-----", Att_s1_t12t2[0, :, :])  # (64,7,7)
        # global visnum
        # for bnum in range(0, 10):
        #     drawmatrix(Att_s1_t12t2[bnum], visnum)
        #     visnum = visnum + 1


        Att_s1_ori = A_keleton2skeleton_s1 * 0.5
        Att_s1_t12t2 = self.softmax(Att_s1_t12t2 + Att_s1_ori)

        N2 = N // 2
        x_s1_t1t2 = x_s1.permute(0, 3, 2, 1).contiguous().view(N2, 2, V, -1)
        x_s1_t1 = x_s1_t1t2[:, 0, :, :]  # [64, 8, 2560]
        x_s1_t2 = x_s1_t1t2[:, 1, :, :]
        x_s1_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t12t2, x_s1_t1))  # t1 to t2  #[64,8,2560]
        x_s1_t2_afterinter = x_s1_t2_interactive + x_s1_t2  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t2_afterinter------", x_s1_t2_afterinter.shape)

        # t2 to t1

        Att_s1_t22t1 = torch.matmul(self.weight_t22t1, x_s1_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s1_t22t1 = self.relu(Att_s1_t22t1)
        Att_s1_ori = A_keleton2skeleton_s1 * 0.5
        Att_s1_t22t1 = self.softmax(Att_s1_t22t1 + Att_s1_ori)

        N2 = N // 2
        x_s1_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t22t1, x_s1_t2))  # t1 to t2  #[64,8,2560]
        x_s1_t1_afterinter = x_s1_t1_interactive + x_s1_t1  # after interactive   #(64,8,2560) bs =128

        x_s1_t1t2_afterinter = torch.cat((x_s1_t1_afterinter, x_s1_t2_afterinter), 0)  # (128,8,2560)
        x_s1_t1t2_afterinter = x_s1_t1t2_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, V)
        # print("x_s1_t1t2_afterinter------", x_s1_t1t2_afterinter.shape)

        x_s1_t2t1_afterinter = torch.cat((x_s1_t2_afterinter, x_s1_t1_afterinter), 0)  # (128,8,2560)
        x_s1_t2t1_afterinter = x_s1_t2t1_afterinter.permute(0, 2, 1).contiguous().view(N, d, T, V)

        '''
              cross-skeleton
              hs2,k *w *hs1,j.T + 0.5* A(k,i)
        '''
        Att = torch.matmul(x_s1_att, self.weight)
        Att = torch.matmul(Att, x_s2_att.permute(0, 2, 1))  # [64, 8, 3]
        Att = self.softmax(Att )

        x_s2 = x_s2_t1t2_afterinter.permute(0,3,2,1).contiguous().view(N,W,-1)                                                     # [128, 3, 40, 64] -> [64, 3, 2560]
        x_s1_glb = torch.einsum('nvw, nwd->nvd', (Att, x_s2))                                                     # [128, 8, 2560]
        x_s1_glb = x_s1_glb.contiguous().view(N, V, -1, d).permute(0,3,2,1)                                            # [128, 8, 2560] -> [128, 8, 40, 64] -> [128, 64, 40, 8]


        # cross S2 t2 to  S1 t1
        Att_s2t2tos1t1 = torch.matmul(x_s1_att, self.weight2)
        # print("S1_to_S2_weight-----", self.weight)

        Att_s2t2tos1t1 = torch.matmul(Att_s2t2tos1t1, x_s2_att_t2t1.permute(0, 2, 1))  # [64*2, 3, 8]
        # print(Att.shape)
        Att_s2t2tos1t1 = self.softmax(Att_s2t2tos1t1)  # 【128，3，8】
        # print(Att.shape)

        x_s2 = x_s2_t2t1_afterinter.permute(0, 3, 2, 1).contiguous().view(N, W,-1)  # [128, 8, 40, 64] -> [128, 8, 2560]
        x_s1_s2t2tos1t1 = torch.einsum('nwv, nvd->nwd', (Att_s2t2tos1t1, x_s2))  # [128, 3, 2560]
        # print("S1_to_S2_x_s2_glb-----", x_s2_glb.shape)
        x_s1_s2t2tos1t1 = x_s1_s2t2tos1t1.contiguous().view(N, V, -1, d).permute(0, 3, 2, 1)

        x_s1_glb = x_s1_glb + x_s1_s2t2tos1t1

        return x_s1_glb, x_s1_t1t2_afterinter





class interskeleton_interaction(nn.Module):   #1280 t0 120
    '''
            fc_k = interskeleton_interaction(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                 t_stride=(1, 2), t_padding=1, weight=(256, 256))

    '''
    def __init__(self,t,  n_j1, n_j2, n_p1, n_p2, t_kernel, t_stride, t_padding, weight):
        super().__init__()


        self.time_conv = nn.Sequential(nn.Conv2d(n_j1, n_j1//2, kernel_size=(t_kernel, 1), stride=(t_stride[1], 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_j1//2),
                                                 nn.Dropout(0, inplace=True))

        self.softmax = nn.Softmax(dim=-1)

        self.weight_t12t2 = nn.Parameter(torch.Tensor(7, 1280))
        self.weight_t22t1 = nn.Parameter(torch.Tensor(7, 1280))

        self.relu = nn.ReLU(inplace=True)
        # nn.init.constant_(self.weight, 0)
        # nn.init.constant_(self.weight2, 1)
        # print(self.weight.shape)


    def forward(self, x_s1, relrec_s1, relsend_s1, relrec_speed, relsend_speed):                                                           # x_s1: [64, 32, 49, 26]

        N, d, T, V = x_s1.size()

        x_s1_att = self.time_conv(x_s1)                            # [64, 64, 10, 7]
        # print("interskeleton_interaction",x_s1_att.shape)


        A_keleton2skeleton_s1 = global_A_keleton2skeleton_s1.cuda(x_s1.get_device())
        '''
        # inter-skeleton ---S1
        '''
        x_s1__att_t1t2 = x_s1_att.contiguous().view(-1, 2, V, 640)
        # x_s1__att_t1t2 = x_s1_att.contiguous().view(-1, 2, V, 2560)
        x_s1_att_t1 = x_s1__att_t1t2[:,0,:,:]  #[N,7,640]
        x_s1_att_t2 = x_s1__att_t1t2[:,1,:,:]

        #t1 to t2

        x_s1_att_t1t2_cat = torch.cat((x_s1_att_t1, x_s1_att_t2), 2)    #(N, 7, 1280)
        Att_s1_t12t2 = torch.matmul(self.weight_t12t2, x_s1_att_t1t2_cat.permute(0,2,1))    #[7,7]
        Att_s1_t12t2 = self.relu(Att_s1_t12t2)
        Att_s1_ori = A_keleton2skeleton_s1 * 0.5


        Att_s1_t12t2 = self.softmax(Att_s1_t12t2 +Att_s1_ori)
        N2 = N //2
        x_s1_t1t2 = x_s1.permute(0, 3, 2, 1).contiguous().view(N2, 2,  V, -1)
        x_s1_t1 = x_s1_t1t2[:, 0, :, :]  #[64, 8, 2560]
        x_s1_t2 = x_s1_t1t2[:, 1, :, :]
        x_s1_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t12t2, x_s1_t1))   #t1 to t2  #[64,8,2560]
        x_s1_t2_afterinter = x_s1_t2_interactive + x_s1_t2   # after interactive   #(64,8,2560) bs =128

        # t2 to t1
        Att_s1_t22t1 = torch.matmul(self.weight_t22t1, x_s1_att_t1t2_cat.permute(0, 2, 1))  # [7,7]
        Att_s1_t22t1 = self.relu(Att_s1_t22t1)
        Att_s1_ori = A_keleton2skeleton_s1 * 0.5

        Att_s1_t22t1 = self.softmax(Att_s1_t22t1+Att_s1_ori)
        N2 = N // 2
        x_s1_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t22t1, x_s1_t2))  # t1 to t2  #[64,8,2560]
        x_s1_t1_afterinter = x_s1_t1_interactive + x_s1_t1  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t1_afterinter------", x_s1_t1_afterinter.shape)

        x_s1_t1t2_afterinter = torch.cat((x_s1_t2_afterinter, x_s1_t1_afterinter), 0) #(128,8,2560)
        # print("x_s1_t1t2_afterinter------", x_s1_t1t2_afterinter.shape)
        x_s1_t1t2_afterinter = x_s1_t1t2_afterinter.permute(0, 2,1).contiguous().view(N,d,T,V)

        return x_s1_t1t2_afterinter







class interskeleton_interaction_S2(nn.Module):  #120 -1280 nodes=3
    '''
           fc_k = interskeleton_interaction(_T, n_j1=dim, n_j2=(640, 128), n_p1=dim, n_p2=(640, 128), t_kernel=3,
                                                 t_stride=(1, 2), t_padding=1, weight=(256, 256))
    '''
    def __init__(self,t,  n_j1, n_j2, n_p1, n_p2, t_kernel, t_stride, t_padding, weight):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.weight = nn.Parameter(torch.Tensor(256, 256))
        self.weight_t12t2 = nn.Parameter(torch.Tensor(120, 120))
        self.weight_t22t1 = nn.Parameter(torch.Tensor(120, 120))

        self.weight2 = nn.Parameter(torch.Tensor(3, 3))
        nn.init.constant_(self.weight, 0)
        nn.init.constant_(self.weight2, 1)


        self.fc = nn.Linear(64 * 40, 120)
        self.fc2 = nn.Linear(120, 64 * 40)

    def forward(self, x_s1, x_s1_att, relrec_s1, relsend_s1, relrec_speed, relsend_speed):                                                           # x_s1: [64, 32, 49, 26]
        # print("S1_to_S2_relrec_s1-----", relrec_s1.shape, relsend_s1.shape, relrec_s2.shape, relsend_s2.shape)                              #[650, 26], [650, 26], [90, 10],[90, 10]
        N, d, T, V = x_s1.size()


        '''
        # inter-skeleton ---S1
        '''
        x_s1_att = self.fc(x_s1_att)
        x_s1__att_t1t2 = x_s1_att.contiguous().view(-1, 2, V, 120)
        x_s1_att_t1 = x_s1__att_t1t2[:,0,:,:]
        x_s1_att_t2 = x_s1__att_t1t2[:,1,:,:]

        #t1 to t2
        Att_s1_t12t2 = torch.matmul(x_s1_att_t2, self.weight_t12t2)
        Att_s1_t12t2 = torch.matmul(Att_s1_t12t2, x_s1_att_t1.permute(0,2,1))                                                #[64, 8, 8]

        # Att_2 = A_joint2part * 0.5
        Att_s1_t12t2 = self.softmax(Att_s1_t12t2)
        N2 = N //2
        x_s1_t1t2 = x_s1.permute(0, 3, 2, 1).contiguous().view(N2, 2,  V, -1)
        x_s1_t1 = x_s1_t1t2[:, 0, :, :]  #[64, 8, 2560]
        x_s1_t2 = x_s1_t1t2[:, 1, :, :]
        x_s1_t2_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t12t2, x_s1_t1))   #t1 to t2  #[64,8,2560]
        x_s1_t2_afterinter = x_s1_t2_interactive + x_s1_t2   # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t2_afterinter------", x_s1_t2_afterinter.shape)

        # t2 to t1
        Att_s1_t22t1 = torch.matmul(x_s1_att_t1, self.weight_t22t1)
        Att_s1_t22t1 = torch.matmul(Att_s1_t22t1, x_s1_att_t2.permute(0, 2, 1))  # [64, 8, 8]

        Att_s1_t22t1 = self.softmax(Att_s1_t22t1)
        N2 = N // 2
        x_s1_t1_interactive = torch.einsum('nwv, nvd->nwd', (Att_s1_t22t1, x_s1_t2))  # t1 to t2  #[64,8,2560]
        x_s1_t1_afterinter = x_s1_t1_interactive + x_s1_t1  # after interactive   #(64,8,2560) bs =128
        # print("x_s1_t1_afterinter------", x_s1_t1_afterinter.shape)

        x_s1_t1t2_afterinter = torch.cat((x_s1_t2_afterinter, x_s1_t1_afterinter), 0) #(128,8,2560)
        x_s1_t1t2_afterinter = x_s1_t1t2_afterinter.permute(0, 2,1).contiguous().view(N,d,T,V)
        # print("x_s1_t1t2_afterinter------", x_s1_t1t2_afterinter.shape)

        x_s1_t1t2_afterinter = x_s1_t1t2_afterinter.permute(0, 3, 2, 1).contiguous().view(N, V, -1)
        x_s1_t1t2_afterinter = self.fc2(x_s1_t1t2_afterinter)  # [N,3,64*40]

        return x_s1_t1t2_afterinter