import numpy as np
import math
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torchvision import models

from collections import Counter
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

## Transform 2D
from custom_transform2D import CustomResize
from custom_transform2D import CustomToTensor

from AD_Dataset import AD_Dataset
from AD_Standard_2DSlicesData import AD_Standard_2DSlicesData
from AD_Standard_2DRandomSlicesData import AD_Standard_2DRandomSlicesData
from AD_Standard_2D3AxisImage import AD_Standard_2D3AxisImage
from AD_Standard_2DRandom3AxisImage import AD_Standard_2DRandom3AxisImage


from AlexNet2D import alexnet
from DensNet import densenet121
from ResNet2D import BinaryResnet
from DenseNet2D import DenseNet121


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Deep Learning AD Predition")



#### Revised 2018 Nov 19 AW
parser.add_argument("--model", default = "alexnet",
                    help="Neural network model")
parser.add_argument("--pretrained", default = "True",
                    help="If using pretrained model weights")
parser.add_argument("--savename", default = "None",
                    help="Surfix for saved file name")
parser.add_argument("--approach", default = "majority_vote",
                    help="Choose if use majrotiy vote approach")
#-- Revised end 2018 Nov 19 AW
parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="AlexNet",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size for training. (default=8)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


# feel free to add more arguments as you need


def main(options):
    # Path configuration
    TRAINING_PATH = 'train_2classes.txt'
    TESTING_PATH = 'test_2classes.txt'
    IMG_PATH = './Image'

    trg_size = (224, 224)

    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])

    if options.approach == "majority_vote":
        dset_train = AD_Standard_2DRandomSlicesData(IMG_PATH, TRAINING_PATH, transformations)
        dset_test = AD_Standard_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)
    elif options.approach == "random":        
        dset_train = AD_Standard_2DRandom3AxisImage(IMG_PATH, TRAINING_PATH, transformations)
        dset_test = AD_Standard_2D3AxisImage(IMG_PATH, TESTING_PATH, transformations)
    elif options.approach == "fixed":   
        dset_train = AD_Standard_2D3AxisImage(IMG_PATH, TRAINING_PATH, transformations)
        dset_test = AD_Standard_2D3AxisImage(IMG_PATH, TESTING_PATH, transformations)

    # Use argument load to distinguish training and testing
    if options.load is None:
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True
                                  )
    else:
        # Only shuffle the data when doing training
        train_loader = DataLoader(dset_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True
                                  )

    test_loader = DataLoader(dset_test,
                             batch_size=options.batch_size,
                             shuffle=False,
                             num_workers=4,
                             drop_last=True
                             )

    use_cuda = torch.cuda.is_available()
    if options.gpuid:
        cuda.set_device(options.gpuid[0])

    #### Revised 2018 Nov 19 AW
    # Initial the model
    #model = alexnet(pretrained=True)

    if options.model == 'default' or options.model == 'alexnet':
        model = alexnet(pretrained=options.pretrained)
    elif options.model == 'densenet121':
        m = models.densenet121(pretrained=options.pretrained)
        model = DenseNet121(m)
        #model = densenet121(pretrained=options.pretrained)
    elif options.model == 'resnet18':
        m = models.resnet18(pretrained=options.pretrained)
        model = BinaryResnet(m)
    else:
        raise ValueError('Unknown model type: ' + options.model)
    #-- Revised end 2018 Nov 19 AW


    # model.load_state_dict(torch.load(options.load))

    if use_cuda:
        model.cuda()
    else:
        model.cpu()

    # Binary cross-entropy loss
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()

    lr = options.learning_rate
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99, last_epoch=-1)

    best_accuracy = float("-inf")

    train_acu_f = open(options.savename + "_train_accuracy.txt", "w")
    train_loss_f = open(options.savename + "_train_loss.txt", "w")
    test_acu_f = open(options.savename + "_test_accuracy.txt", "w")
    test_loss_f = open(options.savename + "_test_loss.txt", "w")

    for epoch_i in range(options.epochs):
        scheduler.step()
        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss, correct_cnt = train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f)
        train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
        train_avg_acu = float(correct_cnt.data.item()) / len(dset_train)

        logging.info(
            "Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data.item(), epoch_i))
        logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))

        if options.approach == "majority_vote":
            correct_cnt = validate_mvote(model, test_loader, use_cuda, criterion, test_loss_f)
        else:
            correct_cnt = validate(model, test_loader, use_cuda, criterion, test_loss_f)
        dev_avg_acu = float(correct_cnt) / len(dset_test)
        logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))

        # write validation accuracy to file
        train_acu_f.write("{0:.5f}\n".format(train_avg_acu))
        test_acu_f.write("{0:.5f}\n".format(dev_avg_acu))

        # if dev_avg_acu > best_accuracy:
        #     best_accuracy = dev_avg_acu
        #     torch.save(model.state_dict(), open(options.save, 'wb'))

    train_acu_f.close()
    train_loss_f.close()
    test_acu_f.close()
    test_loss_f.close()



def train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f):
    # main training loop
    train_loss = 0.0
    correct_cnt = 0.0
    model.train()

    for it, train_data in enumerate(train_loader):
        dt_correct_cnt = 0.0
        dt_train_loss = 0.0
        for data_dic in train_data:
            imgs = Variable(data_dic['image'])
            lbl = data_dic['label']
            ground_truth = Variable(lbl).long()
            if use_cuda:
                imgs, ground_truth = imgs.cuda(), ground_truth.cuda()
            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, ground_truth)
            dt_train_loss += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            dt_correct_cnt += correct_this_batch
            accuracy = float(correct_this_batch.data.item()) / len(ground_truth)
            #logging.info("batch {0} training loss is : {1:.5f}".format(it, loss.data.item()))
            #logging.info("batch {0} training accuracy is : {1:.5f}".format(it, accuracy))
            # write the training loss to file
            train_loss_f.write("{0:.5f}\n".format(loss.data.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        dt_correct_cnt = dt_correct_cnt/len(train_data)
        correct_cnt += dt_correct_cnt
        dt_train_loss = dt_train_loss/len(train_data)
        train_loss += dt_train_loss

    return train_loss, correct_cnt



def validate(model, test_loader, use_cuda, criterion, test_loss_f):
    # validation -- this is a crude estimation because there might be some paddings at the end
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(test_loader):
        ground_truth = test_data[0]['label']
        test_loss = 0.0
        for data_dic in test_data:
            with torch.no_grad():
                imgs = Variable(data_dic['image'])
                if use_cuda:
                    imgs, g_truth = imgs.cuda(), ground_truth.cuda()
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            loss = criterion(test_output, g_truth)
            test_loss += loss
        test_loss = test_loss/len(test_data)
        correct_this_batch = (predict.cpu().squeeze(1) == ground_truth).sum()
        correct_cnt += correct_this_batch
        accuracy =  float(correct_this_batch) / len(ground_truth)
        test_loss_f.write("{0:.5f}\n".format(test_loss))
        logging.info("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

    return correct_cnt


def validate_mvote(model, test_loader, use_cuda, criterion, test_loss_f):
    # validation -- this is a crude estimation because there might be some paddings at the end
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(test_loader):
        vote = []
        ground_truth = test_data[0]['label']
        test_loss = 0.0
        for data_dic in test_data:
            with torch.no_grad():
                imgs = Variable(data_dic['image'])
                if use_cuda:
                    imgs, g_truth = imgs.cuda(), ground_truth.cuda()
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            vote.append(predict)
            loss = criterion(test_output, g_truth)
            test_loss += loss
        # Majority Vote
        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        test_loss = test_loss/len(test_data)
        correct_this_batch = (final_vote.cpu().data == ground_truth).sum()
        correct_cnt += correct_this_batch
        accuracy =  float(correct_this_batch) / len(ground_truth)
        test_loss_f.write("{0:.5f}\n".format(test_loss))
        logging.info("batch {0} dev accuracy is : {1:.5f}".format(it, accuracy))

    return correct_cnt





def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)