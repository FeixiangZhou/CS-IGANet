## first MLP--- Add

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp_JpTrans(nn.Module):
    '''
    MLP
            self.mlp1 = Mlp_JpTrans(n_joint2[0], n_joint2[1], n_joint2[1], drop)    (800, 256, 256)
            self.mlp2 = Mlp_JpTrans(n_joint2[1]*2, n_joint2[1], n_joint2[1], drop)  (256*2, 256, 256 )
            self.mlp3 = Mlp_JpTrans(n_joint2[1]*2, n_joint2[1], n_joint2[1], drop, out_act=False)   (256*2, 256, 256 )

            n_j1=32, n_j2=(800, 256), n_p1=32, n_p2=(800, 256)
    '''

    def __init__(self, n_in, n_hid, n_out, do_prob=0.5, out_act=True):                              # 800, 256,  256
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid+n_in, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.init_weights()
        self.out_act = out_act

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x):
        x_skip = x                                                                                  #[64, 26, 800]
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(torch.cat((x,x_skip),-1))
        x = self.batch_norm(x)
        x = self.leaky_relu(x) if self.out_act==True else x         # [64, 26, 256]
        return x


# class PartLocalInform(nn.Module):

#     def __init__(self, n_in, n_hid, s_kernel, t_kernel, t_stride, t_padding, drop=0.5):
#         super().__init__()
#         self.space_conv = nn.Sequential(nn.Conv2d(n_in, n_hid, kernel_size=(1, s_kernel), bias=True),
#                                         nn.BatchNorm2d(n_hid),
#                                         nn.ReLU(inplace=True))
#         self.time_conv = nn.Sequential(nn.Conv2d(n_hid, n_hid, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True))

#     def forward(self, x):
#         x_space = self.space_conv(x)
#         x_time = self.time_conv(x_space)
#         return x_time

class Partmap2joint(nn.Module):

    def __init__(self):
        super().__init__()

        self.head = [0, 1, 2]
        self.body = [3, 4, 5]
        self.tail = [6]

    def forward(self, part):
        N, d, T, w = part.size()  # [64, 64, 40, 3]
        x = part.new_zeros((N, d, T, 7))

        x[:,:,:,self.head] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.body] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.tail] = part[:,:,:,2].unsqueeze(-1)
        return x





class PartLocalInform(nn.Module):

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

    def forward(self, part):
        N, d, T, w = part.size()  # [64, 256, 7, 10]
        x = part.new_zeros((N, d, T, 26))

        x[:,:,:,self.left_leg_up] = torch.cat((part[:,:,:,0].unsqueeze(-1), part[:,:,:,0].unsqueeze(-1)),-1)
        x[:,:,:,self.left_leg_down] = torch.cat((part[:,:,:,1].unsqueeze(-1), part[:,:,:,1].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_up] = torch.cat((part[:,:,:,2].unsqueeze(-1), part[:,:,:,2].unsqueeze(-1)),-1)
        x[:,:,:,self.right_leg_down] = torch.cat((part[:,:,:,3].unsqueeze(-1), part[:,:,:,3].unsqueeze(-1)),-1)
        x[:,:,:,self.torso] = torch.cat((part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1), part[:,:,:,4].unsqueeze(-1)),-1)
        x[:,:,:,self.head] = torch.cat((part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1), part[:,:,:,5].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_up] = torch.cat((part[:,:,:,6].unsqueeze(-1),part[:,:,:,6].unsqueeze(-1)),-1)
        x[:,:,:,self.left_arm_down] = torch.cat((part[:,:,:,7].unsqueeze(-1), part[:,:,:,7].unsqueeze(-1), part[:,:,:,7].unsqueeze(-1), part[:,:,:,7].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_up] = torch.cat((part[:,:,:,8].unsqueeze(-1), part[:,:,:,8].unsqueeze(-1)),-1)
        x[:,:,:,self.right_arm_down] = torch.cat((part[:,:,:,9].unsqueeze(-1), part[:,:,:,9].unsqueeze(-1), part[:,:,:,9].unsqueeze(-1), part[:,:,:,9].unsqueeze(-1)),-1)

        return x


class BodyLocalInform(nn.Module):

    def __init__(self):
        super().__init__()

        self.torso = [8,9,10,11,12,13]
        self.left_leg = [0,1,2,3]
        self.right_leg = [4,5,6,7]
        self.left_arm = [14,15,16,17,18,19]
        self.right_arm = [20,21,22,23,24,25]

    def forward(self, body):
        N, d, T, w = body.size()  # [64, 256, 7, 10]
        x = body.new_zeros((N, d, T, 26))

        x[:,:,:,self.left_leg] = torch.cat((body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1], body[:,:,:,0:1]),-1)
        x[:,:,:,self.right_leg] = torch.cat((body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,1:2], body[:,:,:,2:3]),-1)
        x[:,:,:,self.torso] = torch.cat((body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3], body[:,:,:,2:3]),-1)
        x[:,:,:,self.left_arm] = torch.cat((body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4], body[:,:,:,3:4]),-1)
        x[:,:,:,self.right_arm] = torch.cat((body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5], body[:,:,:,4:5]),-1)

        return x
        

def node2edge(x, rel_rec, rel_send):            # #[64,26,256]
    # print("node2edge-----", rel_send.shape)

    receivers = torch.matmul(rel_rec, x)                #[650,26]* [64,26,256]  =[64, 650, 256]

    senders = torch.matmul(rel_send, x)                 #[650, 26] [64, 26, 256] =[64, 650, 256]
    distance = receivers - senders
    edges = torch.cat([receivers, distance], dim=2)         ## [64, 650, 512]
    return edges

def edge2node_mean(x, rel_rec, rel_send):   ## [64, 650, 256]
    incoming = torch.matmul(rel_rec.t(), x)  # [26, 650] *[64, 650, 256]= [64, 26, 256]
    nodes = incoming/incoming.size(1)           # [64, 26, 256]
    return nodes


class S1AttInform(nn.Module):
    '''
              self.j2p_1 = S1_to_S2(n_j1=32, n_j2=(1280, 256), n_p1=32, n_p2=(1280, 256), t_kernel=5, t_stride=(1,2), t_padding=2)
              self.embed_s1 = S1AttInform(n_j1, n_j2, t_stride[1], t_kernel, t_padding, drop=0.2, nmp=True)
    '''

    def __init__(self, t, n_joint1, n_joint2,  t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=False):
        super().__init__()
        n_joint1_down = n_joint1 // 2
        self.T = t
        self.time_conv = nn.Sequential(nn.Conv2d(n_joint1, n_joint1_down, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0), bias=True),
                                                 nn.BatchNorm2d(n_joint1_down),
                                                 nn.Dropout(drop, inplace=True))



        if nmp==True:
            self.mlp1 = Mlp_JpTrans(n_joint2[0], n_joint2[1], n_joint2[1], drop)# 640 to 128
            self.mlp2 = Mlp_JpTrans(n_joint2[1]*2, n_joint2[1], n_joint2[1], drop)


            if t==40:
                # self.mlp1_speed = Mlp_JpTrans(n_joint1_down*7, 128, 128, drop)  # speed  njoints=7
                # self.mlp2_speed = Mlp_JpTrans(256, 128, 128, drop)
                # self.mlp3_speed = Mlp_JpTrans(256, 126, 126, drop, out_act=False)
                # self.mlp4_speed = Mlp_JpTrans(128 + 360, 128, 128, drop)   #20 * 18 = 360

                self.mlp1_speed = Mlp_JpTrans(n_joint1_down*7, 126, 126, drop)  # speed  njoints=7
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(252, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 360, 128, 128, drop)   #20 * 18 = 360


            if t == 20:
                # self.mlp1_speed = Mlp_JpTrans(n_joint1_down*7, 128, 128, drop)  # speed  njoints=7, T20
                # self.mlp2_speed = Mlp_JpTrans(256, 128, 128, drop)
                # self.mlp3_speed = Mlp_JpTrans(256, 126, 126, drop, out_act=False)
                # self.mlp4_speed = Mlp_JpTrans(128 + 180, 128, 128, drop)    # 10 * 18 = 180


                self.mlp1_speed = Mlp_JpTrans(n_joint1_down*7, 126, 126, drop)  # speed  njoints=7, T20
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(252, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 180, 128, 128, drop)    # 10 * 18 = 180


            if t == 10:
                # self.mlp1_speed = Mlp_JpTrans(n_joint1_down * 7, 128, 128, drop)  # speed  njoints=7, T20
                # self.mlp2_speed = Mlp_JpTrans(256, 128, 128, drop)
                # self.mlp3_speed = Mlp_JpTrans(256, 126, 126, drop, out_act=False)
                # self.mlp4_speed = Mlp_JpTrans(128 + 90, 128, 128, drop)  # 5 * 18 = 180

                self.mlp1_speed = Mlp_JpTrans(n_joint1_down * 7, 126, 126, drop)  # speed  njoints=7, T20
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(252, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 90, 128, 128, drop)  # 5 * 18 = 180



        else:

            self.mlp1 = Mlp_JpTrans(n_joint2[0], n_joint2[1], n_joint2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send, relrec_speed, relsend_speed):                                           # x: [64, 32, 49, 26], [650,26],[650,26]
        N, D, T, V = x.size()
        # print("S1AttInform_x-----", x[0, 0, 0, :])
        # print(self.layer1)                                                              #False
        x_ = x if self.layer1==True else self.time_conv(x)                            # [64, 64, 10, 7]
        # print("S1AttInform-----", x_.shape)
        x_ = x_.permute(0,2,3,1)                                                      #[ [64,10,7, 64]
        # print("S1AttInform_x_-----", x_[0, 0:2, :, 0])

        x_1 = x_.contiguous().view(N,V,-1)                                             # #[64, 7, 640]
        # print("S1AttInform-----", x_.shape)

        x_2 = x_.contiguous().view(N, self.T//2, -1)                                   # speed   #mouse[64,10,448]
        # print("S1AttInform-----", x_2.shape)

        x_node = self.mlp1(x_1)                                                          #[64,7,128]
        x_node_speed = self.mlp1_speed(x_2)                                             #speed (64,10, 128)
        # print("S1AttInform_x_node-----", x_node.shape)
        if self.nmp==True:
            x_node_skip = x_node
            x_node_skip_speed = x_node_speed
            x_edge = node2edge(x_node, rel_rec, rel_send)                               # [64, 650, 512]
            # print("S1AttInform_x_edge-----", x_edge.shape)
            x_edge_speed = node2edge(x_node_speed, relrec_speed, relsend_speed)                            #speed [64, 600, 512]

            x_edge = self.mlp2(x_edge)                                                  # [64, 650, 256],  mouse--[64,8*7, 256]
            x_edge_speed = self.mlp2_speed(x_edge_speed)                                                    #speed [64, 600, 256]  mouse--[64,20*19, 256]

            # print("S1AttInform_x_edge2-----", x_edge.shape)
            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                          # mouse--[64,7, 128]
            # print("S1AttInform_x_node-----", x_node.shape)

            x_node_speed = edge2node_mean(x_edge_speed, relrec_speed, relsend_speed)                        # speed   mouse--[64,10, 128]


            #### sum
            x_node = x_node + x_node_skip  #[64, 7, 128]
            x_node_speed = x_node_speed + x_node_skip_speed   #speed [ 64, 10, 126]

            x_node_speed = x_node_speed.contiguous().view(N,  self.T//2, -1, V).permute(0,3,2,1)                        #speed [64, 7, 18, 10]
            x_node_speed = x_node_speed.contiguous().view(N, V, -1)                                                     # speed [64, 7, 180]
            # print("S1AttInform_x_node ", x_node_speed.shape)
            x_node =  torch.cat((x_node_speed, x_node), -1)                                                   #speed [64, 7, 128+180]  combine speed and bone
            x_node = self.mlp4_speed(x_node)
        return x_node



        

class S2AttInform(nn.Module):

    def __init__(self, t, n_part1, n_part2, t_stride, t_kernel, t_padding, drop=0.2, layer1=False, nmp=False):
        super().__init__()
        n_part1_down = n_part1 // 2
        self.T = t
        self.time_conv = nn.Sequential(nn.Conv2d(n_part1, n_part1_down, kernel_size=(t_kernel, 1), stride=(t_stride, 1), padding=(t_padding, 0),bias=True),
                                       nn.BatchNorm2d(n_part1_down), nn.Dropout(drop, inplace=True))


        if nmp == True:
            self.mlp1 = Mlp_JpTrans(n_part2[0], n_part2[1], n_part2[1], drop)
            self.mlp2 = Mlp_JpTrans(n_part2[1] * 2, n_part2[1], n_part2[1], drop)
            # self.mlp3 = Mlp_JpTrans(n_part2[1] * 2, n_part2[1], n_part2[1], drop, out_act=False)

            # self.mlp1_speed = Mlp_JpTrans(192, 256, 256, drop)
            # self.mlp2_speed = Mlp_JpTrans(512, 256, 256, drop)
            # self.mlp3_speed = Mlp_JpTrans(512, 255, 255, drop, out_act=False)
            # self.mlp4_speed = Mlp_JpTrans(256 + 1700, 256, 256, drop)


            if t ==40:
                self.mlp1_speed = Mlp_JpTrans(n_part1_down * 3, 126, 126, drop)  # speed  njoints=7
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(252, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 840, 128, 128, drop)  # 20 * 42 = 360


            if t == 20:
                self.mlp1_speed = Mlp_JpTrans(n_part1_down*3, 126, 126, drop)
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(256, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 420, 128, 128, drop)   #10*42
            if t == 10:
                self.mlp1_speed = Mlp_JpTrans(n_part1_down * 3, 126, 126, drop)
                self.mlp2_speed = Mlp_JpTrans(252, 126, 126, drop)
                # self.mlp3_speed = Mlp_JpTrans(256, 126, 126, drop, out_act=False)
                self.mlp4_speed = Mlp_JpTrans(128 + 210, 128, 128, drop)  # 5*42



        else:
            self.mlp1 = Mlp_JpTrans(n_part2[0], n_part2[1], n_part2[1], drop, out_act=False)
        self.init_weights()
        self.layer1 = layer1
        self.nmp = nmp

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, x, rel_rec, rel_send, relrec_speed, relsend_speed):  # x: [64, 32, 49, 26], [650,26],[650,26]
        N, D, T, V = x.size()
        x_ = x if self.layer1 == True else self.time_conv(x)  # [64,32,25,10] mouse[64,64,20,3]
        x_ = x_.permute(0, 2, 3, 1)  #    [64,20,3,64]

        x_1 = x_.contiguous().view(N, V, -1)  # [64,3, 1280]

        x_2 = x_.contiguous().view(N, self.T//2, -1)                                                   # speed T=20  [64,20,192]

        x_node = self.mlp1(x_1)                                                                 # [64,3,256]
        x_node_speed = self.mlp1_speed(x_2)                                                     # speed (64,20, 256)
        if self.nmp == True:
            x_node_skip = x_node
            x_node_skip_speed = x_node_speed
            x_edge = node2edge(x_node, rel_rec, rel_send)                                           # [64, 3*2, 512]
            x_edge_speed = node2edge(x_node_speed, relrec_speed, relsend_speed)                     # speed [64, 20*19, 512]

            x_edge = self.mlp2(x_edge)                                                              # [64, 6, 256]
            x_edge_speed = self.mlp2_speed(x_edge_speed)                                                 # speed [64, 380, 256]

            x_node = edge2node_mean(x_edge, rel_rec, rel_send)                                           # [64, 3, 256]
            x_node_speed = edge2node_mean(x_edge_speed, relrec_speed, relsend_speed)                     # speed [64, 20, 256]

            #### sum
            x_node = x_node + x_node_skip  # [64, 3, 128]
            x_node_speed = x_node_speed + x_node_skip_speed  # speed [ 64, 10, 126]

            x_node_speed = x_node_speed.contiguous().view(N,  self.T//2, -1, V).permute(0, 3, 2, 1)                  # speed [64, 3, 85, 20]
            x_node_speed = x_node_speed.contiguous().view(N, V, -1)                                          # speed [64, 3, 1700]
            x_node = torch.cat((x_node_speed, x_node), -1)                                               # speed [64, 3, 1700+256]  combine speed and bone
            x_node = self.mlp4_speed(x_node)
        return x_node
