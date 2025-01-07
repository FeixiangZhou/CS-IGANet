import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


dir_train_eachvideo = "/training_process_eachvideo.csv"
dir_val_eachvideo = "/validate_process_eachvideo.csv"

dir_train = "/training_process.csv"
dir_val = "/validate_process.csv"

numvideo =0

# dir_train_eachvideo = "/training_process_eachvideo2.csv"
# dir_val_eachvideo = "/validate_process_eachvideo2.csv"
#
# dir_train = "/training_process2.csv"
# dir_val = "/validate_process2.csv"

def saveepochcheckpont( epoch, net, netname):
    #     print('Saving checkpoint...')
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     if not os.path.isdir('checkpoint/' + netname):
    #         os.mkdir('checkpoint/' + netname)
        state = {
            'net': net.state_dict(),
            'epoch': epoch,

        }
        torch.save(state,  netname + '/epoch_{}checkpoint.ckpt'.format(epoch))
        # highPCK = meanPCK
        # config['lowrmse'] = meanPCK






def caculate_acc(gt, predict, result_path=None):
    '''

    :param gt: append every bach
    :param predict: append every batch
    :return: acc
    '''
    global numvideo

    numvideo = numvideo+1
    # print("gt--", len(gt))   #batch
    gt_all = gt[0].numpy().tolist()
    # print(gt_all)
    predict_all = predict[0].numpy().tolist()
    for i in range(len(gt) - 1):
        gt_all = gt_all + gt[i + 1].numpy().tolist()   #combine list
        predict_all = predict_all + predict[i + 1].numpy().tolist()
    # print("gt_all---", len(gt_all))  # all video combine
    # print("predict_all---", len(predict_all))  # all video combine

    predict_all = np.array(predict_all)
    gt_all = np.array(gt_all)


    #test
    dir_label_gt = "label/label_gt%d_my.csv" % (numvideo-1)
    dir_label_predict = "label/label_predict%d_my.csv" % (numvideo-1)

    # np.savetxt(result_path + dir_label_gt, gt_all, delimiter=",")
    # np.savetxt(result_path + dir_label_predict, predict_all, delimiter=",")

    # remove other class
    # index = np.where(gt_all == 7)       # CRIM13(other-11),   RI(solitary-7)
    # print("index------", index)
    # gt_all_noother = np.delete(gt_all, index)
    # predict_all_noother = np.delete(predict_all, index)
    # print("gt, predict---", gt_all_noother.shape, predict_all_noother.shape)
    print("gt, predict---", gt_all.shape, predict_all.shape)


    #save result
    # np.savetxt(result_path + "/result_allvideo_label.csv", gt_all,  delimiter=",")
    # np.savetxt(result_path + "/result_allvideo_predict.csv", predict_all, delimiter=",")

    acc_eachclass, aveacc = calculate_aveacc_CRIM13(gt_all, predict_all) #-----------------------------------CRIM13
    # acc_eachclass, aveacc = calculate_aveacc_PDMB(gt_all, predict_all)  # -----------------------------------PDMB
    # acc_eachclass, aveacc = calculate_aveacc_Rat(gt_all, predict_all) #------------------------------------Rat
    # acc_eachclass, aveacc = calculate_aveacc_RI(gt_all, predict_all) #------------------------------------RI
    acc_top1 = np.mean(predict_all == gt_all)     #top 1


    # acc_eachclass_noother, aveacc_noother = calculate_aveacc_RI(gt_all_noother, predict_all_noother)
    # acc_top1_nother = np.mean(predict_all_noother == gt_all_noother)  # top 1


    # print("acc_all---", acc_all)
    print("acc_all---", acc_eachclass)
    return acc_top1, aveacc, acc_eachclass
    # return acc_top1_nother, aveacc_noother, acc_eachclass_noother


def calculate_aveacc_CRIM13(gt_content,recog_content):
    totalmouse = [0] * 12
    correctmouse = [0] * 12
    for i in range(len(gt_content)):

        if gt_content[i] == 0:      #'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:      #'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:      #'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:      #'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:      #'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:      #'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:      #'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:      #'clean':
            totalmouse[7] += 1
        # if gt_content[i] == 'human':
        #     totalmouse[8] += 1
        if gt_content[i] == 8:      #'sniff':
            totalmouse[8] += 1
        if gt_content[i] == 9:      #'up':
            totalmouse[9] += 1
        if gt_content[i] == 10:     #'walk_away':
            totalmouse[10] += 1
        if gt_content[i] == 11:     #'other':
            totalmouse[11] += 1

        if gt_content[i] == recog_content[i] == 0:
            correctmouse[0] += 1
        if gt_content[i] == recog_content[i] == 1:
            correctmouse[1] += 1
        if gt_content[i] == recog_content[i] == 2:
            correctmouse[2] += 1
        if gt_content[i] == recog_content[i] == 3:
            correctmouse[3] += 1
        if gt_content[i] == recog_content[i] == 4:
            correctmouse[4] += 1
        if gt_content[i] == recog_content[i] == 5:
            correctmouse[5] += 1
        if gt_content[i] == recog_content[i] == 6:
            correctmouse[6] += 1
        if gt_content[i] == recog_content[i] == 7:
            correctmouse[7] += 1
        # if gt_content[i] == recog_content[i] == 'human':
        #     correctmouse[8] += 1
        if gt_content[i] == recog_content[i] == 8:
            correctmouse[8] += 1
        if gt_content[i] == recog_content[i] == 9:
            correctmouse[9] += 1
        if gt_content[i] == recog_content[i] == 10:
            correctmouse[10] += 1
        if gt_content[i] == recog_content[i] == 11:
            correctmouse[11] += 1
    print(totalmouse)

    for m in range(12):
        if totalmouse[m] ==0:
            correctmouse[m] =10
            totalmouse[m] = 1


    acc_eachclass = np.true_divide(correctmouse, totalmouse)

    acclist= []

    for k in range(12):
        if(acc_eachclass[k]<=1):
            acclist.append(acc_eachclass[k])


    aveacc = np.mean(acclist)

    return acc_eachclass, aveacc


def save_result_multiclass_CRIM13(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch','lr', 'acc_top1', 'aveacc', 'approach', 'attack', 'copulation', 'chase', 'circle', 'drink', 'eat',
            'clean', 'sniff', 'up', 'walk_away', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0,  'aveacc': 0, 'approach': 0, 'attack': 0, 'copulation': 0, 'chase': 0, 'circle': 0, 'drink': 0, 'eat': 0,
            'clean': 0, 'sniff': 0, 'up': 0, 'walk_away': 0, 'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['approach'] = acc[0]
    dict['attack'] = acc[1]
    dict['copulation'] = acc[2]
    dict['chase'] = acc[3]
    dict['circle'] = acc[4]
    dict['drink'] = acc[5]
    dict['eat'] = acc[6]
    dict['clean'] = acc[7]
    dict['sniff'] = acc[8]
    dict['up'] = acc[9]
    dict['walk_away'] = acc[10]
    dict['other'] = acc[11]

    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train

    else:
        dir = dir_val

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)

def save_result_multiclass_CRIM13_everyvideo(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch','lr', 'acc_top1', 'aveacc', 'approach', 'attack', 'copulation', 'chase', 'circle', 'drink', 'eat',
            'clean', 'sniff', 'up', 'walk_away', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0,  'aveacc': 0, 'approach': 0, 'attack': 0, 'copulation': 0, 'chase': 0, 'circle': 0, 'drink': 0, 'eat': 0,
            'clean': 0, 'sniff': 0, 'up': 0, 'walk_away': 0, 'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['approach'] = acc[0]
    dict['attack'] = acc[1]
    dict['copulation'] = acc[2]
    dict['chase'] = acc[3]
    dict['circle'] = acc[4]
    dict['drink'] = acc[5]
    dict['eat'] = acc[6]
    dict['clean'] = acc[7]
    dict['sniff'] = acc[8]
    dict['up'] = acc[9]
    dict['walk_away'] = acc[10]
    dict['other'] = acc[11]

    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train_eachvideo

    else:
        dir = dir_val_eachvideo

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)


def calculate_numberofsample(gt_content):
    totalmouse = [0] * 12
    for i in range(len(gt_content)):
        # gt_content[i] = gt_content[i].strip()
        if gt_content[i] == 0:  # 'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:  # 'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:  # 'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:  # 'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:  # 'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:  # 'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:  # 'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:  # 'clean':
            totalmouse[7] += 1
        if gt_content[i] == 8:  # 'sniff':
            totalmouse[8] += 1
        if gt_content[i] == 9:  # 'up':
            totalmouse[9] += 1
        if gt_content[i] == 10:  # 'walk_away':
            totalmouse[10] += 1
        if gt_content[i] == 11:  # 'other':
            totalmouse[11] += 1

    sampleweight = [1. / i if i != 0 else 0 for i in totalmouse]

    # sampleweight[-1] = 0   #adjust weight, remove other----
    # sampleweight[7] = 0  # adjust weight, remove other---- rat/RI
    # sampleweight[-1] = sampleweight[-1] *2
    # sampleweight[6] = sampleweight[6] * 0.5  # eat
    # sampleweight[-1] = sampleweight[-1] * 1.2  # other
    return sampleweight






def calculate_numberofsample_Rat(gt_content):
    totalmouse = [0] * 8
    for i in range(len(gt_content)):
        # gt_content[i] = gt_content[i].strip()
        if gt_content[i] == 0:  # 'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:  # 'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:  # 'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:  # 'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:  # 'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:  # 'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:  # 'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:  # 'clean':
            totalmouse[7] += 1

    sampleweight = [1. / i if i != 0 else 0 for i in totalmouse]

    # sampleweight[-1] = 0   #adjust weight, remove other----
    # sampleweight[7] = 0  # adjust weight, remove other---- rat

    return sampleweight

def calculate_aveacc_Rat(gt_content,recog_content):
    totalmouse = [0] * 8
    correctmouse = [0] * 8
    for i in range(len(gt_content)):

        if gt_content[i] == 0:      #'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:      #'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:      #'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:      #'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:      #'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:      #'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:      #'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:      #'clean':
            totalmouse[7] += 1


        if gt_content[i] == recog_content[i] == 0:
            correctmouse[0] += 1
        if gt_content[i] == recog_content[i] == 1:
            correctmouse[1] += 1
        if gt_content[i] == recog_content[i] == 2:
            correctmouse[2] += 1
        if gt_content[i] == recog_content[i] == 3:
            correctmouse[3] += 1
        if gt_content[i] == recog_content[i] == 4:
            correctmouse[4] += 1
        if gt_content[i] == recog_content[i] == 5:
            correctmouse[5] += 1
        if gt_content[i] == recog_content[i] == 6:
            correctmouse[6] += 1
        if gt_content[i] == recog_content[i] == 7:
            correctmouse[7] += 1


    for m in range(8):
        if totalmouse[m] ==0:
            correctmouse[m] =10
            totalmouse[m] = 1


    acc_eachclass = np.true_divide(correctmouse, totalmouse)

    acclist= []

    for k in range(8):
        if(acc_eachclass[k]<=1):
            acclist.append(acc_eachclass[k])

    aveacc = np.mean(acclist)

    return acc_eachclass, aveacc


def save_result_multiclass_Rat(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch','lr', 'acc_top1', 'aveacc', 'attack', 'lateral_threat', 'sniff', 'submission', 'boxing', 'approach', 'avoidance',
            'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0,  'aveacc': 0, 'attack': 0, 'lateral_threat': 0, 'sniff': 0, 'submission': 0, 'boxing': 0, 'approach': 0, 'avoidance': 0,
            'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['attack'] = acc[0]
    dict['lateral_threat'] = acc[1]
    dict['sniff'] = acc[2]
    dict['submission'] = acc[3]
    dict['boxing'] = acc[4]
    dict['approach'] = acc[5]
    dict['avoidance'] = acc[6]
    dict['other'] = acc[7]


    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train

    else:
        dir = dir_val

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)

def save_result_multiclass_Rat_everyvideo(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch', 'lr', 'acc_top1', 'aveacc', 'attack', 'lateral_threat', 'sniff', 'submission', 'boxing',
               'approach', 'avoidance',
               'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0, 'aveacc': 0, 'attack': 0, 'lateral_threat': 0, 'sniff': 0,
            'submission': 0, 'boxing': 0, 'approach': 0, 'avoidance': 0,
            'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['attack'] = acc[0]
    dict['lateral_threat'] = acc[1]
    dict['sniff'] = acc[2]
    dict['submission'] = acc[3]
    dict['boxing'] = acc[4]
    dict['approach'] = acc[5]
    dict['avoidance'] = acc[6]
    dict['other'] = acc[7]

    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train_eachvideo

    else:
        dir = dir_val_eachvideo

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)





def calculate_numberofsample_RI(gt_content):
    totalmouse = [0] * 9
    for i in range(len(gt_content)):
        # gt_content[i] = gt_content[i].strip()
        if gt_content[i] == 0:  # 'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:  # 'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:  # 'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:  # 'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:  # 'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:  # 'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:  # 'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:  # 'clean':
            totalmouse[7] += 1
        if gt_content[i] == 8:  # 'clean':
            totalmouse[8] += 1

    sampleweight = [1. / i if i != 0 else 0 for i in totalmouse]

    # sampleweight[-1] = 0   #adjust weight, remove other----
    # sampleweight[7] = 0  # adjust weight, remove other---- rat
    # sampleweight[-1] = sampleweight[-1] *2
    return sampleweight

def calculate_aveacc_RI(gt_content,recog_content):
    totalmouse = [0] * 9
    correctmouse = [0] * 9
    for i in range(len(gt_content)):

        if gt_content[i] == 0:      #'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:      #'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:      #'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:      #'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:      #'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:      #'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:      #'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:      #'clean':
            totalmouse[7] += 1
        if gt_content[i] == 8:  # 'clean':
            totalmouse[8] += 1

        if gt_content[i] == recog_content[i] == 0:
            correctmouse[0] += 1
        if gt_content[i] == recog_content[i] == 1:
            correctmouse[1] += 1
        if gt_content[i] == recog_content[i] == 2:
            correctmouse[2] += 1
        if gt_content[i] == recog_content[i] == 3:
            correctmouse[3] += 1
        if gt_content[i] == recog_content[i] == 4:
            correctmouse[4] += 1
        if gt_content[i] == recog_content[i] == 5:
            correctmouse[5] += 1
        if gt_content[i] == recog_content[i] == 6:
            correctmouse[6] += 1
        if gt_content[i] == recog_content[i] == 7:
            correctmouse[7] += 1
        if gt_content[i] == recog_content[i] == 8:
            correctmouse[8] += 1

    for m in range(9):
        if totalmouse[m] ==0:
            correctmouse[m] =10
            totalmouse[m] = 1


    acc_eachclass = np.true_divide(correctmouse, totalmouse)

    acclist= []

    for k in range(9):
        if(acc_eachclass[k]<=1):
            acclist.append(acc_eachclass[k])

    aveacc = np.mean(acclist)

    return acc_eachclass, aveacc


def save_result_multiclass_RI(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch','lr', 'acc_top1', 'aveacc', 'Allogrooming', 'Approaching', 'Following', 'Moving away', 'attacking', 'pining', 'nose contact',
            'solitary', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0,  'aveacc': 0, 'Allogrooming': 0, 'Approaching': 0, 'Following': 0, 'Moving away': 0, 'attacking': 0, 'pining': 0, 'nose contact': 0,
            'solitary': 0,  'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['Allogrooming'] = acc[0]
    dict['Approaching'] = acc[1]
    dict['Following'] = acc[2]
    dict['Moving away'] = acc[3]
    dict['attacking'] = acc[4]
    dict['pining'] = acc[5]
    dict['nose contact'] = acc[6]
    dict['solitary'] = acc[7]
    dict['other'] = acc[8]


    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train

    else:
        dir = dir_val

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)

def save_result_multiclass_RI_everyvideo(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch', 'lr', 'acc_top1', 'aveacc', 'Allogrooming', 'Approaching', 'Following', 'Moving away',
               'attacking', 'pining', 'nose contact',  'solitary', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0, 'aveacc': 0, 'Allogrooming': 0, 'Approaching': 0, 'Following': 0,
            'Moving away': 0, 'attacking': 0, 'pining': 0, 'nose contact': 0,
            'solitary': 0, 'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['Allogrooming'] = acc[0]
    dict['Approaching'] = acc[1]
    dict['Following'] = acc[2]
    dict['Moving away'] = acc[3]
    dict['attacking'] = acc[4]
    dict['pining'] = acc[5]
    dict['nose contact'] = acc[6]
    dict['solitary'] = acc[7]
    dict['other'] = acc[8]

    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train_eachvideo

    else:
        dir = dir_val_eachvideo

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)



def calculate_numberofsample_PDMB(gt_content):
    totalmouse = [0] * 9
    for i in range(len(gt_content)):
        # gt_content[i] = gt_content[i].strip()
        if gt_content[i] == 0:  # 'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:  # 'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:  # 'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:  # 'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:  # 'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:  # 'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:  # 'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:  # 'clean':
            totalmouse[7] += 1
        if gt_content[i] == 8:  # 'clean':
            totalmouse[8] += 1

    sampleweight = [1. / i if i != 0 else 0 for i in totalmouse]

    # sampleweight[-1] = 0   #adjust weight, remove other----
    # sampleweight[7] = 0  # adjust weight, remove other---- rat
    # sampleweight[-1] = sampleweight[-1] *2
    return sampleweight

def calculate_aveacc_PDMB(gt_content,recog_content):
    totalmouse = [0] * 9
    correctmouse = [0] * 9
    for i in range(len(gt_content)):

        if gt_content[i] == 0:      #'approach'
            totalmouse[0] += 1
        if gt_content[i] == 1:      #'attack':
            totalmouse[1] += 1
        if gt_content[i] == 2:      #'copulation':
            totalmouse[2] += 1
        if gt_content[i] == 3:      #'chase':
            totalmouse[3] += 1
        if gt_content[i] == 4:      #'circle':
            totalmouse[4] += 1
        if gt_content[i] == 5:      #'drink':
            totalmouse[5] += 1
        if gt_content[i] == 6:      #'eat':
            totalmouse[6] += 1
        if gt_content[i] == 7:      #'clean':
            totalmouse[7] += 1
        if gt_content[i] == 8:  # 'clean':
            totalmouse[8] += 1

        if gt_content[i] == recog_content[i] == 0:
            correctmouse[0] += 1
        if gt_content[i] == recog_content[i] == 1:
            correctmouse[1] += 1
        if gt_content[i] == recog_content[i] == 2:
            correctmouse[2] += 1
        if gt_content[i] == recog_content[i] == 3:
            correctmouse[3] += 1
        if gt_content[i] == recog_content[i] == 4:
            correctmouse[4] += 1
        if gt_content[i] == recog_content[i] == 5:
            correctmouse[5] += 1
        if gt_content[i] == recog_content[i] == 6:
            correctmouse[6] += 1
        if gt_content[i] == recog_content[i] == 7:
            correctmouse[7] += 1
        if gt_content[i] == recog_content[i] == 8:
            correctmouse[8] += 1

    for m in range(9):
        if totalmouse[m] ==0:
            correctmouse[m] =10
            totalmouse[m] = 1


    acc_eachclass = np.true_divide(correctmouse, totalmouse)

    acclist= []

    for k in range(9):
        if(acc_eachclass[k]<=1):
            acclist.append(acc_eachclass[k])

    aveacc = np.mean(acclist)

    return acc_eachclass, aveacc


def save_result_multiclass_PDMB(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch','lr', 'acc_top1', 'aveacc', 'approach', 'chase', 'circle', 'eat', 'clean', 'sniff', 'up',
            'walk_away', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0,  'aveacc': 0, 'approach': 0, 'chase': 0, 'circle': 0, 'eat': 0, 'clean': 0, 'sniff': 0, 'up': 0,
            'walk_away': 0,  'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['approach'] = acc[0]
    dict['chase'] = acc[1]
    dict['circle'] = acc[2]
    dict['eat'] = acc[3]
    dict['clean'] = acc[4]
    dict['sniff'] = acc[5]
    dict['up'] = acc[6]
    dict['walk_away'] = acc[7]
    dict['other'] = acc[8]


    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train

    else:
        dir = dir_val

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)

def save_result_multiclass_PDMB_everyvideo(epoch, lr, acc, aveacc, acc_top1, results_dir, istraining = True):
    columns = ['epoch', 'lr', 'acc_top1', 'aveacc', 'approach', 'chase', 'circle', 'eat', 'clean', 'sniff', 'up',
               'walk_away', 'other']
    dict = {'epoch': 0, 'lr': 0, 'acc_top1': 0, 'aveacc': 0, 'approach': 0, 'chase': 0, 'circle': 0, 'eat': 0,
            'clean': 0, 'sniff': 0, 'up': 0,
            'walk_away': 0, 'other': 0}
    dict['epoch'] = epoch
    dict['lr'] = lr
    dict['acc_top1'] = acc_top1
    dict['aveacc'] = aveacc

    dict['approach'] = acc[0]
    dict['chase'] = acc[1]
    dict['circle'] = acc[2]
    dict['eat'] = acc[3]
    dict['clean'] = acc[4]
    dict['sniff'] = acc[5]
    dict['up'] = acc[6]
    dict['walk_away'] = acc[7]
    dict['other'] = acc[8]

    df = pd.DataFrame([dict])

    if istraining:
        dir = dir_train_eachvideo

    else:
        dir = dir_val_eachvideo

    if epoch == 0:
        df.to_csv(results_dir + dir, columns=columns)
    else:
        df.to_csv(results_dir + dir, mode='a', header=False, columns=columns)



# matrixpath = "work_dir/crim13/matriximg/approach/"  #7163
# matrixpath = "work_dir/crim13/matriximg/chase/"      #779
matrixpath = "work_dir/crim13/matriximg/walkaway/"  #7008

# matrixpath = "work_dir/crim13/matriximg/all/"

# matrixpath = "work_dir/crim13/matriximg/epoch0/approach/"
# matrixpath = "work_dir/crim13/matriximg/epoch0/chase/"
# matrixpath = "work_dir/crim13/matriximg/epoch0/walkaway/"





def drawmatrix(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'intert12t2/txtfile/intermatrixt12t2_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'intert12t2/intermatrixt12t2_%d.png' % num)

def drawmatrix2(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'intert22t1/txtfile/intermatrixt22t1_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'intert22t1/intermatrixt22t1_%d.png' % num)


def drawmatrix3(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'crosssame/txtfile/crossmatrixsame_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'crosssame/crossmatrixsame_%d.png' % num)

def drawmatrix4(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'crossdiff/txtfile/crossmatrixdiff_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'crossdiff/crossmatrixdiff_%d.png' % num)

def drawmatrix5(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'intrat1/txtfile/intramatrix_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'intrat1/intramatrix_%d.png' % num)

def drawmatrix6(a, num):
    a = a.cpu().data.numpy()
    np.savetxt(matrixpath+'intrat2/txtfile/intramatrix_%d.txt' % num, a, delimiter=",")

    matfig = plt.figure(figsize=(6, 6))
    ax = plt.matshow(a, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.hot, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.Reds, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlGnBu, fignum=matfig.number)
    # ax = plt.matshow(array1, cmap=plt.cm.YlOrBr, fignum=matfig.number)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.colorbar(ax.colorbar, fraction=0.045)

    plt.savefig(matrixpath+'intrat2/intramatrix_%d.png' % num)


# visimgpath = "work_dir/crim13/visimg/CS/11/"  # 7008

# visimgpath = "work_dir/crim13/visimg/IAT/11/"  # 7008
#
#
# if not os.path.isdir(visimgpath):
#     os.mkdir(visimgpath)


class FeatureVisualization():
    def __init__(self,fea, num):
        self.feamap=fea
        self.num = num

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.feamap
        # print(input.shape) #(1, 3, 224, 224))
        return input

    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape) #[1, 128, 112, 112]) conv featuremaps
        feature =features.mean(0).mean(0)


        # feature=features[0,0,:,:]
        print(feature.shape) #([1, 112, 112])

        # feature=feature.view(feature.shape[1],feature.shape[2])
        # print(feature.shape) #(112,112)

        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.cpu().data.numpy()

        #use sigmod to [0,1]
        # feature= 1.0/(1+np.exp(-1*feature)) # to [0,255]
        feature = (feature - np.amin(feature)) / (np.amax(feature) - np.amin(feature) + 1e-5)
        # print(feature)
        feature=np.round(feature*255)
        # print(feature[0])
        feature = np.asarray(feature, dtype=np.uint8)
        # feature = cv2.applyColorMap(feature, cv2.COLORMAP_HOT)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        cv2.imwrite(visimgpath+'feamaps%d.png' % self.num , feature)




def save_outfeature(feature, results_dir, numvideo):
    outfeaturepath = 'outfeature_%d.csv' %(numvideo-1)
    with open (results_dir+outfeaturepath, 'ab') as f:
        feature = feature.cpu().data.numpy()
        np.savetxt(f, feature, delimiter=",")
        # np.savetxt(results_dir+'outfeature.csv', feature, delimiter=",")


