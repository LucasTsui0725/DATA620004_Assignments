import os
import sys
import argparse
import warnings
from datetime import datetime
cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(cur_dir)
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from utils import *
from models import ResNet18
from dataset import CIFAR_100_Dataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

def train_model(model, data_loader, optimizer, loss_func, device, metrics=None):
    model.train()
    epoch_loss = 0
    num = 0
    predict_tensor = torch.tensor([]).to(device)
    for idx, (data, label) in enumerate(data_loader):
        batch_size, height, width, in_channel = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
        data = data.type(torch.FloatTensor).transpose(1, 3).to(device)
        label = label.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        predict = model(data)
        predict_label = predict.max(dim=1)[1]
        predict_tensor = torch.concat([predict_tensor, predict_label])
        loss = loss_func(predict, label)
        epoch_loss += loss.item() * batch_size
        num += batch_size
        loss.backward()
        optimizer.step()

    return epoch_loss / num, predict_tensor

def predict_model(model, data_loader, loss_func, device, metrics=None):
    model.eval()
    epoch_loss = 0
    num = 0
    predict_tensor = torch.tensor([]).to(device)
    label_tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        for idx, (data, label) in enumerate(data_loader):
            batch_size, height, width, in_channel = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
            data = data.type(torch.FloatTensor).transpose(1, 3).to(device)
            label = label.type(torch.FloatTensor).to(device)
            
            predict = model(data)
            predict_label = predict.max(dim=1)[1]
            loss = loss_func(predict, label)
            epoch_loss += loss.item() * batch_size
            num += batch_size
            predict_tensor = torch.concat([predict_tensor, predict_label])
            label_tensor = torch.concat([label_tensor, torch.max(label, 1)[1]])
    if metrics is None:
        return epoch_loss / num, predict_tensor
    else:
        return epoch_loss / num, predict_tensor, metrics(predict_tensor, label_tensor)

# cutmix
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./dataset/cifar100")
    parser.add_argument("--augment", type=str, default=None, choices=['cutmix', 'mixup', 'cutout'])
    parser.add_argument("--prob", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--input-channel", type=int)
    parser.add_argument("--output-channel", type=int)
    parser.add_argument("--num-epoch", type=int, default=150)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--metrics", type=str, default="accuracy", nargs="+", choices=["precision", "recall", "f1_score", "micro_f1", "macro_f1", "accuracy"])

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--print-freq", type=int, default=5)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    seed_all(args.seed)
    if args.augment is None:
        file_path = f"lr_{args.lr}_weight_decay_{args.weight_decay}_aug_{args.augment}"
    else:
        file_path = f"lr_{args.lr}_weight_decay_{args.weight_decay}_aug_{args.augment}_prob_{args.prob}_beta_{args.beta}"
    if args.log_path is None:
        log_path = os.path.join(os.getcwd(), 'log', file_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    else:
        log_path = args.log_path
    writer = SummaryWriter(log_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.0203, 0.1994, 0.2010])])
    train_valid_dataset = datasets.cifar.CIFAR100(args.data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.cifar.CIFAR100(args.data_path, train=False, transform=transform, download=True)
    train_dataset, valid_dataset, train_label, valid_label= train_test_split(train_valid_dataset.data, train_valid_dataset.targets, test_size=0.2, stratify=train_valid_dataset.targets)
    if args.augment is not None:
        aug_train_dataset = CIFAR_100_Dataset(train_dataset / 255, train_label, shuffle=True, prob=args.prob, augment=args.augment, beta=args.beta)
    else:
        aug_train_dataset = CIFAR_100_Dataset(train_dataset / 255, train_label, shuffle=True)
    train_dataset = CIFAR_100_Dataset(train_dataset / 255, train_label)
    valid_dataset = CIFAR_100_Dataset(valid_dataset / 255, valid_label)
    test_dataset = CIFAR_100_Dataset(test_dataset.data / 255, test_dataset.targets)

    aug_train_loader = DataLoader(aug_train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    resnet18_model = ResNet18(input_channel=args.input_channel, output_channel=args.output_channel).to(args.gpu)
    optimizer = Adam(params=resnet18_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = F.cross_entropy
    early_stopping = EarlyStopping(verbose=args.verbose, patience=10)
    if isinstance(args.metrics, str):
        args.metrics = [args.metrics]
    metrics = Metrics(metrics=args.metrics)

    print("Save model parameters: {}".format(log_path))
    for epoch in range(args.num_epoch):
        aug_train_loss, aug_train_predict = train_model(model=resnet18_model, data_loader=aug_train_loader, optimizer=optimizer, loss_func=loss_func, device=args.gpu)
        train_loss, train_predict, train_metrics = predict_model(model=resnet18_model, data_loader=train_loader, loss_func=loss_func, device=args.gpu, metrics=metrics)
        valid_loss, valid_predict, valid_metrics = predict_model(model=resnet18_model, data_loader=valid_loader, loss_func=loss_func, device=args.gpu, metrics=metrics)

        train_acc = train_metrics['accuracy']
        valid_acc = valid_metrics['accuracy']
        writer.add_scalars("Loss Visualization", {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch)
        early_stopping(val_acc=valid_acc, model=resnet18_model, model_path=log_path)
        if epoch % args.print_freq == 0:
            print(f"EPOCH: {epoch}\tTRAIN LOSS: {train_loss:.4f}\tTRAIN ACC: {train_acc:.4f}\tVALID LOSS: {valid_loss:.4f}\tVALID ACC: {valid_acc:.4f}")
        if early_stopping.early_stop:
            print("Early Stop")
            break
    writer.close()

    best_model_path = os.path.join(log_path, 'model_param.pth')
    resnet18_model.load_state_dict(torch.load(best_model_path))
    test_loss, test_predict, test_metrics = predict_model(model=resnet18_model, data_loader=test_loader, loss_func=loss_func, device=args.gpu, metrics=metrics)
    print(f"Test Dataset: {test_metrics}")
