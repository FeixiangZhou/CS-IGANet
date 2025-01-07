import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import time

sys.path.extend(['../'])
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        # print("labelpath---------", self.label_path)
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
                print(self.sample_name)
                print(self.label)

        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:128]
            self.data = self.data[0:128]
            self.sample_name = self.sample_name[0:128]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        sample = self.sample_name[index]
        # print("1------", data_numpy.shape)  #(3, 300, 18, 2)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)     # sample time sequence  300 to 150 (3, 150, 18, 2)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)   #(3, 150, 18, 2)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # return data_numpy, label, index, sample
        #
        return data_numpy, label, index

    def top_k(self, score, top_k):
        #print("len label-----", len(self.label))   #4275
        lenscore = score.shape[0]
        rank = score.argsort()   #rank = score.argsort(1)  small to large

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label[0:lenscore])] #[True, False.......]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        # print(sample_name)
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]  #the first data (3,300,18,2)
        # print(data[:2,1,:,0])

        data = data.reshape((1,) + data.shape) #(1,3,300,18,2)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]

            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward

            print(edge)

            pose = []
            for m in range(M):   #2
                a = []
                for i in range(len(edge)):  #17
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)  #len =2

            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(4,8):

                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        # print(v1)
                        x1 = data[0, :2, t, v1, m]

                        # print(x1)
                        x2 = data[0, :2, t, v2, m]
                        # print(v2)
                        # print(x2)

                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            # print("x", data[0, 0, t, [4, 3],m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            # print("y", data[0, 1, t, [4, 3], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])


                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')

                plt.savefig('image/' + str(t) + '.jpg')
                # plt.cla()
                # fig.canvas.flush_events()
                # plt.pause(0.1)
                time.sleep(0.1)





if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    # data_path = "../data/ntu/xview/val_data_joint.npy"
    # label_path = "../data/ntu/xview/val_label.pkl"
    # graph = 'graph.ntu_rgb_d.Graph'
    # test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    data_path = "/scratch/3dpoint/fz64/dataset/kinetics/val_data_joint.npy"
    label_path = "/scratch/3dpoint/fz64/dataset/kinetics/val_label.pkl"
    graph = 'graphori.kinetics.Graph'
    test(data_path, label_path, vid='qYTS6Yn6J_Q', graph=graph)
