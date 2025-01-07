import torch
import torch.nn as nn
import torch.nn.functional as F
from graphrepre.layers_iat import  PMA, MAB
from math import ceil
from graphrepre.parameter import config

class GraphRepresentation(torch.nn.Module):

    def __init__(self, args):

        super(GraphRepresentation, self).__init__()

        self.args = args
        self.num_features = args['num_features']
        self.nhid = args['num_hidden']
        self.num_classes = args['num_classes']
        self.pooling_ratio = args['pooling_ratio']
        self.dropout_ratio = args['dropout']

    def get_pools(self):

        pools = nn.ModuleList([gap])

        return pools



class GraphTransformerEncode(GraphRepresentation):

    def __init__(self, args, A, num_point, Time):

        super(GraphTransformerEncode, self).__init__(args)
        self.ln = args['ln']
        self.num_heads = args['multi_head']
        self.cluster = args['cluster']
        if num_point==7:
            self.model_sequence = args['model_string'].split('-')
        else:
            self.model_sequence = args['model_string'].split('-')

        self.pools = self.get_pools(_A = A, _num_point= num_point, T = Time)
        # self.classifier = self.get_classifier()
        self.A = A


    def forward(self, data, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed):
        batch_x = data
        # For Graph Multiset Transformer
        for _index, _model_str in enumerate(self.model_sequence):

            if _model_str == 'GMPool_G':

                batch_x = self.pools[_index](batch_x,  relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed,  graph=None)    #【1，72，384】, go --PAM forward

            else:

                batch_x = self.pools[_index](batch_x, relrec_s1, relsend_s1,  relrec_s2, relsend_s2,  relrec_speed, relsend_speed)  #【1，72，128】， 【1，1，128】

            extended_attention_mask = None


        ##### For Classification
        # x = self.classifier(x) #[1,2]
        # x = F.log_softmax(x, dim=-1)

        return batch_x

    def get_pools(self, _input_dim=None, reconstruction=False, _A = None, _num_point= 7, T = 40):

        pools = nn.ModuleList()
        # _input_dim = self.nhid * self.args.num_convs if _input_dim is None else _input_dim   #384
        _input_dim = self.args['input_dim']
        _output_dim = self.nhid    #120
        # _num_nodes = 3 if _num_point==8 else 1   #8-3-1 , 3-1

        _num_nodes = ceil(self.pooling_ratio *_num_point)  #8-3-1, 3-2-1
        _num_nodes_2 = _num_nodes
        # print("_num_nodes---------", _num_nodes)

        for _index, _model_str in enumerate(self.model_sequence):

            if (_index == len(self.model_sequence) - 1) and (reconstruction == False):

                _num_nodes = 1

            if _model_str == 'GMPool_G':

                pools.append(
                    PMA(_A, _input_dim, _output_dim, self.num_heads, _num_point, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=self.args['mab_conv'], _model_str = 'GMPool_G', _T =T)   # num_heads =1
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            elif _model_str == 'GMPool_I':

                pools.append(
                    PMA(_A, _output_dim, _input_dim, self.num_heads, _num_nodes_2, _num_nodes, ln=self.ln, cluster=self.cluster, mab_conv=self.args['mab_conv'], _model_str = 'GMPool_I', _T = T)
                )

                _num_nodes = ceil(self.pooling_ratio * _num_nodes)

            # elif _model_str == 'SelfAtt':
            #
            #     pools.append(
            #         SAB(_A, _num_nodes_2, _input_dim, _output_dim, self.num_heads, ln=self.ln, cluster=self.cluster) # 64*40, 120
            #     )

                # _input_dim = _output_dim
                # _output_dim = _output_dim

            else:

                raise ValueError("Model Name in Model String <{}> is Unknown".format(_model_str))

        # pools.append(nn.Linear(_input_dim, self.nhid))

        return pools



class GraphtransformerUpdategraph(nn.Module):
    '''
    update multi graph
    '''
    def __init__(self, A, _num_point, dim_in, dim_out, num_heads=1, ln=True, cluster=False, mab_conv=None):
        super(GraphtransformerUpdategraph, self).__init__()
        self.mab_graph = MAB(A, _num_point, dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv='GCN', _model_str='GMPool_U')

    def forward(self, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask=None, graph=None):
        # return self.mab_liner(X, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask, graph)
        return self.mab_graph(X, X, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed,attention_mask, graph)



class Graphtransformerlabelmodel(nn.Module):
  #Graphtransformerlabelmodel(A, 64, 120, 120)
    def __init__(self, A, _num_point, dim_in, dim_out, num_heads=1, ln=True, cluster=False, mab_conv=None):
        super(Graphtransformerlabelmodel, self).__init__()
        self.mab_graph = MAB(A, _num_point, _num_point, dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv='GCN', _model_str='GMPool_label', _T=1)

    def forward(self, X, relrec_s1=None, relsend_s1=None,  relrec_s2=None, relsend_s2=None, relrec_speed=None, relsend_speed=None, attention_mask=None, graph=None):

        # return self.mab_liner(X, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask, graph)
        return self.mab_graph(X, X, relrec_s1, relsend_s1, relrec_s2, relsend_s2, relrec_speed, relsend_speed,attention_mask, graph)





class GraphTransformerDecode(nn.Module):

    def __init__(self, A, _num_point, dim_in, dim_out, num_heads=1, ln=True, cluster=False, mab_conv=None):
        super(GraphTransformerDecode, self).__init__()
        self.S1 = nn.Parameter(torch.Tensor(3, 1))  # scale :1-----3-----8
        self.S2 = nn.Parameter(torch.Tensor(_num_point, 3))  #scale :1-----3-----8
        self.mab_liner= MAB(A, 1, 3, dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster, conv='GCN', _model_str='GMPool_TCN')
        self.mab_graph = MAB(A, 3,_num_point, dim_out, dim_out, dim_in, num_heads, ln=ln, cluster=cluster, conv='GCN', _model_str='GMPool_GCN')


    def forward(self, X, relrec_s1=None, relsend_s1=None,  relrec_s2=None, relsend_s2=None, relrec_speed=None, relsend_speed=None, attention_mask=None, graph=None):

        Q1 = torch.matmul(self.S1, X)
        scale_liner =  self.mab_liner(Q1, X,  relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask, graph)  #[N, 3, 2560]
        Q2 = torch.matmul(self.S2, scale_liner)
        scale_graph = self.mab_graph(Q2, scale_liner, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed,  attention_mask, graph) #[N,8,64*40]
        return scale_graph


class GraphTransformerDecode_s2(nn.Module):

    def __init__(self, A, _num_point, dim_in, dim_out, num_heads=1, ln=True, cluster=False, mab_conv=None):
        super(GraphTransformerDecode_s2, self).__init__()
        self.S1 = nn.Parameter(torch.Tensor(2, 1))  # scale :1-----2-----3
        self.S2 = nn.Parameter(torch.Tensor(_num_point, 2))  #scale :1-----2-----3
        self.mab_liner = MAB(A, 1, 2, dim_in, dim_in, dim_out, num_heads, ln=ln, cluster=cluster,  conv='GCN', _model_str='GMPool_TCN')
        self.mab_graph = MAB(A, 2, _num_point, dim_out, dim_out, dim_in, num_heads, ln=ln, cluster=cluster, conv='GCN', _model_str='GMPool_GCN')


    def forward(self, X, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask=None, graph=None):

        Q1 = torch.matmul(self.S1, X)
        scale_liner =  self.mab_liner(Q1, X,  relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed, attention_mask, graph)  #[N, 3, 120]
        Q2 = torch.matmul(self.S2, scale_liner)
        scale_graph = self.mab_graph(Q2, scale_liner, relrec_s1, relsend_s1,  relrec_s2, relsend_s2, relrec_speed, relsend_speed,  attention_mask, graph) #[N,8,64*40]
        return scale_graph

