#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from batch_gen import BatchGenerator
from utilis import saveepochcheckpont
from utilis import caculate_acc, save_result_multiclass_CRIM13, save_result_multiclass_CRIM13_everyvideo


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot






class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument(
        '--resume', '-r', action='store_true', help='resume from checkpoint')



#multi-video
    parser.add_argument(
        '--split_train',
        default='/scratch/3dpoint/fz64/dataset/CRIM13/splits/trainfile.txt',
        help='the work folder for storing results')

    parser.add_argument(
        '--split_test',
        default='/scratch/3dpoint/fz64/dataset/CRIM13/splits/testfile.txt',
        help='the work folder for storing results')

    parser.add_argument(
        '--numclass',
        default= 12,
        help='the work folder for storing results')

    parser.add_argument(
        '--sample_rate',
        default=1,
        help='the work folder for storing results')

    parser.add_argument(
        '--features_path',
        default= '/scratch/3dpoint/fz64/dataset/CRIM13/',
        help='the work folder for storing results')

    parser.add_argument(
        '--gt_path',
        default= '/scratch/3dpoint/fz64/dataset/CRIM13/',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        # default='./config/nturgbd-cross-view/test_bone.yaml',
        default='./config/kinetics-skeleton/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--istraining',
        type=str2bool,
        default=True,
        help='if ture, save traingprocess')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=200)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                # if os.path.isdir(arg.model_saved_name):
                    # print('log_dir: ', arg.model_saved_name, 'already exist')
                    # answer = input('delete it? y/n:')
                    # if answer == 'y':
                    #     shutil.rmtree(arg.model_saved_name)
                    #     print('Dir removed: ', arg.model_saved_name)
                    #     input('Refresh the website of tensorboard by pressing any keys')
                    # else:
                    #     print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        # self.load_data_multivideo()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def resume_checkpoint(self, net):
        '''
        :return: 从保存的checkpoint开始训练
        '''
        if self.arg.resume:

            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join(self.arg.model_saved_name, 'epoch_46checkpoint.ckpt'))
            net.load_state_dict(checkpoint['net'])
            self.arg.start_epoch = checkpoint['epoch'] + 1


    def load_data(self):  #ok
        #------
        self.batch_gen = BatchGenerator(self.arg.numclass, self.arg.gt_path, self.arg.features_path, self.arg.sample_rate)
        # batch_gen.read_data(vid_list_file)
        self.batch_gen.read_csvdata(self.arg.split_train)

        self.batch_gen_val = BatchGenerator(self.arg.numclass, self.arg.gt_path, self.arg.features_path,

                                            self.arg.sample_rate)
        self.batch_gen_val.read_csvdata_val(self.arg.split_test)


        Feeder = import_class(self.arg.feeder)  #load class Feeder
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(           #DataLoader-------
                dataset=Feeder(**self.arg.train_feeder_args),   # arg.train_feeder_args  the input of class Feeder
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)

    def load_data_multivideo(self):
        Feeder = import_class(self.arg.feeder)  # load class Feeder
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(  # DataLoader-------
                dataset=Feeder(**self.arg.train_feeder_args),  # arg.train_feeder_args  the input of class Feeder
                batch_size=self.arg.batch_size,
                sampler=self.sampler,
                # shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)

    def load_data_multivideo_TCN(self):
        Feeder = import_class(self.arg.feeder)  # load class Feeder
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(  # DataLoader-------
                dataset=Feeder(**self.arg.train_feeder_args),  # arg.train_feeder_args  the input of class Feeder
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)

    def load_model(self): #ok
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device

        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)  #copy Model's  file

        print(Model)

        self.model = Model(**self.arg.model_args).cuda(output_device)#gpu
        # self.model = Model(**self.arg.model_args).to(device)  #

        # print(self.model)



        self.loss = nn.CrossEntropyLoss().cuda(output_device) #gpu
        self.loss_self = nn.MSELoss().cuda(output_device)
        # self.loss = nn.CrossEntropyLoss().to(device)  #cpu

        if self.arg.weights:  #the weights for network initializatio
            print(" in arg.weights-----------------------------")
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if torch.cuda.is_available():
            if type(self.arg.device) is list: #--------------------------------------------------------------nn.DataParallel
                if len(self.arg.device) > 1:
                    self.model = nn.DataParallel(
                        self.model,
                        device_ids=self.arg.device,
                        output_device=output_device)

        '''        
        
        multi-skeleton
                   
        '''

        V, W = 7, 3
        S = 10  # speed
        S2 = 5
        off_diag_joint, off_diag_part = np.ones([V, V]) - np.eye(V, V), np.ones([W, W]) - np.eye(W, W)
        off_diag_speed = np.ones([S, S]) - np.eye(S, S)
        off_diag_speed2 = np.ones([S2, S2]) - np.eye(S2, S2)

        print("-------------------", self.output_device)
        self.relrec_speed = torch.FloatTensor( np.array(encode_onehot(np.where(off_diag_speed)[1]), dtype=np.float32)).to(self.output_device)  # [380,20] T=20
        self.relsend_speed = torch.FloatTensor( np.array(encode_onehot(np.where(off_diag_speed)[0]), dtype=np.float32)).to(self.output_device)

        self.relrec_speed2 = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_speed2)[1]), dtype=np.float32)).to(self.output_device)  # [380,20] T=20
        self.relsend_speed2 = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_speed2)[0]), dtype=np.float32)).to(self.output_device)

        self.relrec_joint = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_joint)[1]), dtype=np.float32)).to(self.output_device)
        self.relsend_joint = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_joint)[0]), dtype=np.float32)).to(self.output_device)
        self.relrec_part = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_part)[1]), dtype=np.float32)).to(self.output_device)
        self.relsend_part = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag_part)[0]), dtype=np.float32)).to(self.output_device)

    def load_optimizer(self): #ok
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                # self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(      #adjust LR
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch)) #0

    def save_arg(self): #ok
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch): #ok
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step))) #0.01, 0.001
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):

        self.model.train()
        if (epoch > self.arg.only_train_epoch):
            self.model.apply(fix_bn)  # fix batchnorm
        self.print_log('Training epoch-------------------------------: {}'.format(epoch))
        loss_value = []
        self.adjust_learning_rate(epoch)
        numvideo=0
        #multi-video

        gt_allvideo = []
        predict_allvideo = []

        self.batch_gen.read_csvdata(self.arg.split_train)  # --------------------------------shuffle the train list[]
        while(self.batch_gen.has_next()):   # number of videos
            score_frag = []
            gt_everyvideo = []
            predict_everyvideo= []
            numvideo = numvideo + 1
            print("Training epoch | video num: ----------------------------------------------:", epoch, numvideo)

            batch_input_path, batch_target_path, sampler = self.batch_gen.next_batch(batch_size =1)
            self.arg.train_feeder_args['data_path'] = batch_input_path[0]
            self.arg.train_feeder_args['label_path'] = batch_target_path[0]
            self.sampler = sampler
            print(batch_input_path, batch_target_path)

            if (epoch > self.arg.only_train_epoch):
                self.load_data_multivideo_TCN()
            else:
                self.load_data_multivideo()

            # self.load_data_multivideo()

            loader = self.data_loader['train']   # ------------------------------------------------------------ inpute data

            self.train_writer.add_scalar('epoch', epoch, self.global_step)
            self.record_time()
            timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
            process = tqdm(loader)   #-----------------output
            if self.arg.only_train_part:
                if epoch > self.arg.only_train_epoch: #0
                    # print('only train part, require grad')
                    for key, value in self.model.named_parameters():
                        value.requires_grad = False
                        # print("key----", key)
                        if 'labelmodeling' in key:
                            value.requires_grad = True
                            # print(key + '-require grad')
                else:
                    # print('only train part, do not require grad')
                    for key, value in self.model.named_parameters():
                        # value.requires_grad = False
                        if 'labelmodeling' in key:
                            value.requires_grad = False
                            # print(key + '-not require grad')


            for batch_idx, (data, label, index) in enumerate(process):      # read data
                print('\n')
                print("epoch {} batch index {}, 0-12: {}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}".format(epoch, batch_idx, (label == 0).sum(), (label == 1).sum(), (label == 2).sum(),
                                                                                          (label == 3).sum(), (label == 4).sum(), (label == 5).sum(),
                                                                                          (label == 6).sum(), (label == 7).sum(), (label == 8).sum(),
                                                                                          (label == 9).sum(), (label == 10).sum(), (label == 11).sum()))
                self.global_step += 1
                # get data
                data = data.float().cuda(self.output_device)  # data (3,300,18,2)
                # print("data shape :", data.shape)
                # print("train data-----", data[0,:,0,0,0])

                label =label.long().cuda(self.output_device) #[128]
                # print("label shape :", label.shape)

                timer['dataloader'] += self.split_time()


                output, output_TCN, similarity_ori, similarity_l1= self.model(data, self.relrec_joint,  self.relsend_joint,
                                                                               self.relrec_part,
                                                                               self.relsend_part,
                                                                               self.relrec_speed,
                                                                               self.relsend_speed,
                                                                               self.relrec_speed2,
                                                                               self.relsend_speed2)

                # output, output_TCN, x_s1_1, x_s2_1, similarity_ori, similarity_l1 = self.model(data, self.relrec_joint, self.relsend_joint, self.relrec_part, self.relsend_part, self.relrec_speed, self.relsend_speed)
                # print("similarity_ori-------", similarity_ori.shape)
                # print("similarity_l1-------", similarity_l1.shape)
                # output = self.model(data)        #
                # print("train output-----",output

                # if batch_idx == 0 and epoch == 0:
                #     self.train_writer.add_graph(self.model, output)
                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0
                loss = self.loss(output, label) + l1
                loss_TCN = self.loss(output_TCN, label) + l1
                # selfloss_graph = self.loss_self(x_s1_1, x_s2_1)
                selfloss_node = self.loss_self(similarity_ori, similarity_l1)

                if epoch > self.arg.only_train_epoch:
                    # loss = loss
                    loss = loss_TCN                                                   # fix GCN, update the TCN net
                    # loss + loss_TCN + 0.5 * selfloss_graph + 0.5 * selfloss_node  # update the whole net
                else:
                    # loss = loss
                    loss = loss  + 0.5* selfloss_node
                    # loss = loss + 0.1 * selfloss_node
                    # loss = loss + 1* selfloss_node
                    # loss = loss +1.5 * selfloss_node

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_value.append(loss.data.item())
                print("train loss--------", loss.data.item())
                timer['model'] += self.split_time()

                # Acc
                if epoch > self.arg.only_train_epoch:
                    value, predict_label = torch.max(output_TCN.data.cpu(), 1)
                else:
                    value, predict_label = torch.max(output.data.cpu(), 1)
                # value, predict_label = torch.max(output.data.cpu(), 1)


                acc = torch.mean((predict_label == label.data.cpu()).float())

                # combine all videos
                # score_frag.append(output.data.cpu().numpy())
                gt_everyvideo.append(label.data.cpu())
                predict_everyvideo.append(predict_label)
                gt_allvideo.append(label.data.cpu())
                predict_allvideo.append(predict_label)


                self.train_writer.add_scalar('acc', acc, self.global_step)
                self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
                self.train_writer.add_scalar('loss_l1', l1, self.global_step)
                # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

                # statistics
                self.lr = self.optimizer.param_groups[0]['lr']
                self.train_writer.add_scalar('lr', self.lr, self.global_step)
                # if self.global_step % self.arg.log_interval == 0:
                #     self.print_log(
                #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                #             batch_idx, len(loader), loss.data[0], lr))
                timer['statistics'] += self.split_time()

            acc_everyvideo_top1,acc_everyvideo_aveacc,acc_everyvideo_eachclass = caculate_acc(gt_everyvideo, predict_everyvideo)
            print("each video top1, aveacc-----", acc_everyvideo_top1, acc_everyvideo_aveacc)

            save_result_multiclass_CRIM13_everyvideo(epoch, 0, acc_everyvideo_eachclass, acc_everyvideo_aveacc,
                                                     acc_everyvideo_top1, self.arg.work_dir, self.arg.istraining)

        acc_allvideo_top1,acc_allvideo_aveacc,acc_allvideo_eachclass = caculate_acc(gt_allvideo, predict_allvideo)

        self.batch_gen.reset()

        #save training process
        save_result_multiclass_CRIM13(epoch, self.lr , acc_allvideo_eachclass, acc_allvideo_aveacc, acc_allvideo_top1,
                                      self.arg.work_dir, self.arg.istraining)

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        self.print_log(
            '\tacc_allvideo_top1: {:.4f}.'.format(acc_allvideo_top1))
        self.print_log(
            '\tacc_allvideo_aveacc: {:.4f}.'.format(acc_allvideo_aveacc))

        if save_model:
            # state_dict = self.model.state_dict()
            # weights = OrderedDict([[k.split('module.')[-1],
            #                         v.cpu()] for k, v in state_dict.items()])
            # torch.save(weights, self.arg.model_saved_name + 'epoch-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')  #---------save model

            saveepochcheckpont(epoch, self.model,self.arg.model_saved_name)




    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('validate epoch-------------------------------: {}'.format(epoch))
        numvideo = 0
        gt_allvideo = []
        predict_allvideo = []

        # print("model.fc.weight", self.model.module.fc.weight)

        while (self.batch_gen_val.has_next()):  # number of videos
            gt_everyvideo = []
            predict_everyvideo = []
            numvideo = numvideo + 1
            print("validate epoch | video num: ----------------------------------------------:", epoch, numvideo)
            batch_input_path, batch_target_path, sampler = self.batch_gen_val.next_batch(batch_size=1)
            self.arg.test_feeder_args['data_path'] = batch_input_path[0]
            self.arg.test_feeder_args['label_path'] = batch_target_path[0]

            # print(batch_input_path, batch_target_path)

            if (epoch > self.arg.only_train_epoch):
                self.load_data_multivideo_TCN()
            else:
                self.load_data_multivideo()


            for ln in loader_name:
                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                process = tqdm(self.data_loader[ln])
                for batch_idx, (data, label, index) in enumerate(process):
                    with torch.no_grad():
                        data = data.float().cuda(self.output_device)
                        label = label.long().cuda(self.output_device)
                        output,output_TCN, similarity_ori, similarity_l1  = self.model(data,
                                               self.relrec_joint,
                                               self.relsend_joint,
                                               self.relrec_part,
                                               self.relsend_part,
                                               self.relrec_speed,
                                               self.relsend_speed,
                                               self.relrec_speed2,
                                               self.relsend_speed2)
                        # output, output_TCN,  x_s1_1, x_s2_1, similarity_ori, similarity_l1 = self.model(data, self.relrec_joint, self.relsend_joint, self.relrec_part, self.relsend_part, self.relrec_speed, self.relsend_speed)
                        # print("eval output-----", output)

                        # weight = np.array([0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.5])
                        # weight = torch.from_numpy(weight).cuda(self.output_device)

                        output_TCN = torch.softmax(output_TCN, -1)
                        output = torch.softmax(output, -1)

                        output_TCN = 0.1 *output_TCN + output



                        if isinstance(output, tuple):
                            output, l1 = output
                            l1 = l1.mean()
                        else:
                            l1 = 0
                        loss = self.loss(output, label)

                        loss_TCN = self.loss(output_TCN, label)
                        # selfloss_graph = self.loss_self(x_s1_1, x_s2_1)
                        selfloss_node = self.loss_self(similarity_ori, similarity_l1)

                        if epoch > self.arg.only_train_epoch:
                            loss = loss_TCN
                            # loss + loss_TCN + 0.5 * selfloss_graph + 0.5 * selfloss_node  # update the whole net
                        else:
                            # loss = loss
                            loss = loss + 0.5 * selfloss_node
                            # loss = loss + 0.5 * selfloss_graph + 0.5 * selfloss_node

                        score_frag.append(output.data.cpu().numpy())
                        loss_value.append(loss.data.item())
                        # print("val loss--------", loss.data.item())

                        if epoch > self.arg.only_train_epoch:
                            _, predict_label = torch.max(output_TCN.data.cpu(), 1)
                        else:
                            _, predict_label = torch.max(output.data.cpu(), 1)

                        # _, predict_label = torch.max(output.data.cpu(), 1)

                        step += 1

                        # combine all videos
                        # score_frag.append(output.data.cpu().numpy())
                        gt_everyvideo.append(label.data.cpu())
                        predict_everyvideo.append(predict_label)
                        gt_allvideo.append(label.data.cpu())
                        predict_allvideo.append(predict_label)

                    # if wrong_file is not None or result_file is not None:
                    #     predict = list(predict_label.cpu().numpy())
                    #     true = list(label.data.cpu().numpy())
                    #     for i, x in enumerate(predict):
                    #         if result_file is not None:
                    #             f_r.write(str(x) + ',' + str(true[i]) + '\n')
                    #         if x != true[i] and wrong_file is not None:
                    #             f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
                score = np.concatenate(score_frag) #( , 128,400) to (128* n, 400)
                print("eva----score", score.shape)
                loss = np.mean(loss_value)
                accuracy = self.data_loader[ln].dataset.top_k(score, 1)#------------------------------------top_k metric
                if accuracy > self.best_acc:  #0
                    self.best_acc = accuracy
                # self.lr_scheduler.step(loss)
                print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)

                acc_everyvideo_top1,acc_everyvideo_aveacc,acc_everyvideo_eachclass = caculate_acc(gt_everyvideo, predict_everyvideo,
                                              self.arg.work_dir)  #_-------------------------caculate acc
                print("each video top1, aveacc-----", acc_everyvideo_top1, acc_everyvideo_aveacc )
                save_result_multiclass_CRIM13_everyvideo(epoch, 0, acc_everyvideo_eachclass, acc_everyvideo_aveacc, acc_everyvideo_top1, self.arg.work_dir, self.arg.istraining)

                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('loss', loss, self.global_step)
                    self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                    self.val_writer.add_scalar('acc', accuracy, self.global_step)

                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                self.print_log('\t VideoNo {}'.format(numvideo))
                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                for k in self.arg.show_topk: #[1,5]
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

                if save_score:
                    with open('{}/epoch{}_{}_score.pkl'.format(
                            self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                        pickle.dump(score_dict, f)

        acc_allvideo_top1,acc_allvideo_aveacc,acc_allvideo_eachclass = caculate_acc(gt_allvideo, predict_allvideo,  self.arg.work_dir)  # _-------------------------caculate all video acc

        self.batch_gen_val.reset()
        save_result_multiclass_CRIM13(epoch, 0, acc_allvideo_eachclass, acc_allvideo_aveacc,acc_allvideo_top1,self.arg.work_dir, self.arg.istraining)

        self.print_log(
            '\tacc_allvideo_top1: {:.4f}.'.format(acc_allvideo_top1))
        self.print_log(
            '\tacc_allvideo_aveacc: {:.4f}.'.format(acc_allvideo_aveacc))

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            print("len global_step", self.global_step)
            self.resume_checkpoint(self.model)

            # print("model.fc_calcweight_s2.weight", self.model.module.fc_calcweight_s2.weight)
            # print("model.fc.weight", self.model.module.fc.weight)
            # print("model.labelmodeling.conv_out.weight",
            #       self.model.module.labelmodeling.conv_out.weight[0:5, 4, 0])  # (12,64,1)
            # print("model.fc_calcweight_s2.weight.requires_grad:",
            #       self.model.module.fc_calcweight_s2.weight.requires_grad)
            # print("model.labelmodeling.conv_out.requires_grad:",
            #       self.model.module.labelmodeling.conv_out.weight.requires_grad)


            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.arg.istraining = True
                self.train(epoch, save_model=save_model)

                self.arg.istraining = False
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            self.arg.istraining = False
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            # if self.arg.weights is None:
            #     raise ValueError('Please appoint --weights.')
            checkpointepoch =  'epoch_142checkpoint.ckpt'   #------------------------------------------------------------load epoch
            # checkpointepoch = 'checkpoint_freezeeTCNpoech11/epoch_10checkpoint.ckpt'  # ------------------------------------------------------------load epoch

            checkpoint = torch.load(os.path.join(self.arg.model_saved_name, checkpointepoch))
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['net'])

            self.arg.print_log = True
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('epoch: {}.'.format(checkpointepoch))
            self.eval(epoch=start_epoch, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):        #load class
    components = name.split('.') #['model','agcn', 'Model']
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod  #<class 'model.agcn.Model'>


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
