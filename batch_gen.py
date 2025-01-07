import torch
import numpy as np
import random
import pickle
from utilis import calculate_numberofsample, calculate_numberofsample_Rat
from torch.utils.data import WeightedRandomSampler

class BatchGenerator(object):
    def __init__(self, num_classes, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        # self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        # random.shuffle(self.list_of_examples)

    def has_next(self):
        if isinstance(self.list_of_examples, list):
            if self.index < len(self.list_of_examples): #len = 21
                return True
            return False
        else:
            if self.index < 1:
                return True
            else:
                return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]

        # listf = [self.list_of_examples[0]]
        # self.list_of_examples = listf


        print(self.list_of_examples)
        print(len(self.list_of_examples))
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def read_csvdata(self, vid_list_file):
        file_ptr = np.loadtxt(vid_list_file, delimiter=",", dtype=str)
        file_ptr = file_ptr.tolist()
        self.list_of_examples = file_ptr
        # random.shuffle(self.list_of_examples)
        print("train data list---", self.list_of_examples) # len = 0 ,2



    def read_csvdata_val(self, vid_list_file):
        file_ptr = np.loadtxt(vid_list_file, delimiter=",", dtype=str)
        file_ptr = file_ptr.tolist()
        self.list_of_examples = file_ptr
        print("val data list---", self.list_of_examples) # len = 0 ,2



    def next_batch(self, batch_size):

        if isinstance(self.list_of_examples, list):

            batch = self.list_of_examples[self.index:self.index + batch_size]

            # print("batch-----", batch)
            self.index += batch_size

            batch_input = []
            batch_target = []
            data_path = []
            label_path = []
            for vid in batch:
                # print("vid----------", vid)
                data = self.features_path + vid
                # print("data-----", data)
                label = self.gt_path + vid.split('.')[0] + '.pkl'
                # print("label-----", label)

                data_path.append(data)
                label_path.append(label)

                with open(label_path[0], 'rb') as f:
                    sample_name, label = pickle.load(f, encoding='latin1')
                    # print(self.sample_name)
                    # print("input data len:", len(label))

                weight = calculate_numberofsample(label)
                samplesweight = torch.tensor([weight[t] for t in label])  # [120]
                # print(len(samplesweight))

                sampler = WeightedRandomSampler(samplesweight, len(samplesweight), replacement=True)

                return data_path, label_path, sampler
        else:

            batch =  self.list_of_examples

            # print("only one file-----", batch)
            self.index += batch_size
            # print("index-----",  self.index)

            data_path =[]
            label_path =[]
            vid = batch
            data = self.features_path + vid
            # print("data-----", data)
            label = self.gt_path + vid.split('.')[0] + '.pkl'
            # print("label-----", label)
            data_path.append(data)
            label_path.append(label)

            with open(label_path[0],'rb') as f:
                sample_name, label = pickle.load(f, encoding='latin1')
                # print(self.sample_name)
            # print("input data len:", len(label))

            weight = calculate_numberofsample(label) #-------------------------------------------------Rat
            samplesweight = torch.tensor([weight[t] for t in label])  # [120]
            # print(len(samplesweight))

            sampler = WeightedRandomSampler(samplesweight, len(samplesweight), replacement=True)


            return data_path, label_path,sampler






