import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from collections import Counter

from custom_transform2D import CustomResize
from custom_transform2D import CustomToTensor

from AD_Dataset import AD_Dataset
from AD_Standard_2DSlicesData import AD_Standard_2DSlicesData
from AD_Standard_2DRandomSlicesData import AD_Standard_2DRandomSlicesData


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Deep Learning AD Predition.")

parser.add_argument("--network_type", "--nt", default="AlexNet2D",
                    choices=["AlexNet2D", "AlexNet3D", "ResNet2D", "ResNet3D"],
                    help="Deep network type. (default=AlexNet)")
parser.add_argument("--load",
                    help="Load saved network weights.")
parser.add_argument("--save", default="best_model",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=50, type=int,
                    help="Epochs through the data. (default=50)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.0001)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=16, type=int,
                    help="Batch size for training. (default=16)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


# feel free to add more arguments as you need

class newResNet(torch.nn.Module):

    def __init__(self, orig_resnet):
        super().__init__()
        self.orig_resnet = orig_resnet
        self.final_linear = torch.nn.Linear(512, 2)  # TODO
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.orig_resnet.conv1(x)
        x = self.orig_resnet.bn1(x)
        x = self.orig_resnet.relu(x)
        x = self.orig_resnet.maxpool(x)

        x = self.orig_resnet.layer1(x)
        x = self.orig_resnet.layer2(x)
        x = self.orig_resnet.layer3(x)
        x = self.orig_resnet.layer4(x)

        x = self.orig_resnet.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = self.fc(x) # TODO
        x = self.final_linear(x)  # TODO

        return self.logsoftmax(x)


class slicer(nn.Module):
  def __init__(self, batch_size):
    super(slicer, self).__init__()
    self.weight_ax = nn.Parameter(torch.zeros(18,1) + 1/18)
    self.weight_cor = nn.Parameter(torch.zeros(18,1) + 1/18)
    self.weight_sag = nn.Parameter(torch.zeros(18,1) + 1/18)
    self.batch_size = batch_size
  def forward(self, input):
    ax_slice = torch.mm(input[0].resize(self.batch_size*224*224, 18), self.weight_ax).resize(self.batch_size,224,224)
    cor_slice = torch.mm(input[1].resize(self.batch_size*224*224, 18), self.weight_cor).resize(self.batch_size,224,224)
    sag_slice = torch.mm(input[2].resize(self.batch_size*224*224, 18), self.weight_sag).resize(self.batch_size,224,224)
    return torch.stack((ax_slice, cor_slice,sag_slice), dim = 1)

class SlicerNet(torch.nn.Module):

  def __init__(self, orig_resnet):
      super().__init__()
      self.orig_resnet = orig_resnet
      self.slicer = slicer(options.batch_size)

  def forward(self, x):
      x = self.slicer(x)
      x = self.orig_resnet(x)
      return x


def main(options):
    # Path configuration
    TRAINING_PATH = 'train_2classes.txt'
    TESTING_PATH = 'test_2classes.txt'
    IMG_PATH = './Image'

    trg_size = (224, 224)

    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])
    dset_train = AD_Standard_2DRandomSlicesData(IMG_PATH, TRAINING_PATH, transformations)
    dset_test = AD_Standard_2DSlicesData(IMG_PATH, TESTING_PATH, transformations)

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


    # Initial the model
    m = models.resnet18()
    newRes = newResNet(m)
    newRes.load_state_dict(torch.load("./trained_best_model"))
    model = SlicerNet(newRes)

    if use_cuda:
        m.cuda()
        model.cuda()
    else:
        m.cpu()
        model.cpu()

    # Binary cross-entropy loss
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()

    lr = options.learning_rate
    #param = list(model.slicer.parameters()) + list(model.final_linear.parameters())
    optimizer = torch.optim.Adam(model.slicer.parameters(), lr=lr)
    #optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99, last_epoch=-1)

    best_accuracy = float("-inf")

    train_loss_f = open("train_loss.txt", "w")
    test_acu_f = open("test_accuracy.txt", "w")

    for epoch_i in range(options.epochs):

        scheduler.step()
        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss, correct_cnt = train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f)
        # each instance in one batch has 3 views
        train_avg_loss = train_loss / (len(dset_train) / options.batch_size)
        train_avg_acu = float(correct_cnt.data.item()) / len(dset_train)
        logging.info(
            "Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data.item(), epoch_i))
        logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))

        correct_cnt = validate(model, test_loader, use_cuda, criterion)
        dev_avg_acu = float(correct_cnt.data.item()) / 96 #TODO
        logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))
        # write validation accuracy to file
        test_acu_f.write("{0:.5f}\n".format(dev_avg_acu))

        if dev_avg_acu > best_accuracy:
            best_accuracy = dev_avg_acu
            torch.save(model.state_dict(), 'trained_best_model')

    train_loss_f.close()
    test_acu_f.close()

def train(model, train_loader, use_cuda, criterion, optimizer, train_loss_f):
    # main training loop
    train_loss = 0.0
    correct_cnt = 0.0
    model.train()
    for it, train_data in enumerate(train_loader):
        imgs = []
        for data_dic in train_data:
            imgs.append(Variable(data_dic['image']).cuda())
            lbl = data_dic['label']
        ground_truth = Variable(lbl).long()
        if use_cuda:
            ground_truth = ground_truth.cuda()
        train_output = model(imgs)
        _, predict = train_output.topk(1)
        loss = criterion(train_output, ground_truth)
        train_loss += loss
        correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
        correct_cnt += correct_this_batch
        # print(correct_this_batch.data[0])
        accuracy = float(correct_this_batch.data.item()) / len(ground_truth)
        # accuracy = 1
        logging.info("batch {0} training loss is : {1:.5f}".format(it, loss.data.item()))
        logging.info("batch {0} training accuracy is : {1:.5f}".format(it, accuracy))
        # print(loss.data.item(), accuracy)
        # write the training loss to file
        train_loss_f.write("{0:.5f}\n".format(loss.data.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss, correct_cnt



def validate(model, test_loader, use_cuda, criterion):
    # validation -- this is a crude estimation because there might be some paddings at the end
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(test_loader):
        imgs = []
        for data_dic in test_data:
            with torch.no_grad():
                if use_cuda:
                    imgs.append(Variable(data_dic['image']).cuda())
                    lbl = data_dic['label']
                else:
                    imgs.append(Variable(data_dic['image']))
                    lbl = data_dic['label']
        test_output = model(imgs)
        _, predict = test_output.topk(1)
        ground_truth = Variable(lbl).long()
        if use_cuda:
            ground_truth = ground_truth.cuda()
        correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
        correct_cnt += correct_this_batch
        #print(predict, correct_this_batch)
        accuracy = float(correct_this_batch.data.item()) / len(ground_truth)
        #print(correct_this_batch.data.item(), len(ground_truth))
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
