import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def show_img(input, size=(128, 128)):
    if isinstance(input, torch.Tensor):
        input = input.numpy()
    plt.imshow(Image.fromarray(input).resize(size))

class EarlyStopping():
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self._counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
    
    def __call__(self, val_acc, model, model_path=None):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_acc, model, model_path)
        elif score < self.best_score - self.delta:
            self._counter += 1
            if self._counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_acc, model, model_path)
            self._counter = 0
    
    def _save_checkpoint(self, val_acc, model, model_path=None):
        if model_path is None:
            file_path = os.path.join(os.getcwd(), 'model_param.pth')
        else:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            file_path = os.path.join(model_path, 'model_param.pth')
        torch.save(model.state_dict(), file_path)
        if self.verbose:
            print("Validation Acc Increase ({:.6f} --> {:.6f})".format(self.val_acc_max, val_acc))
        self.val_acc_max = val_acc

class Metrics():
    def __init__(self, metrics=None):
        self._metrics = metrics
        self._check_config()

    def _check_config(self):
        default_metrics = ["precision", "recall", "f1_score", "micro_f1", "macro_f1", "accuracy"]
        if isinstance(self._metrics, str):
            self._metrics = [self._metrics]
        for metric in self._metrics:
            if metric.lower() not in default_metrics:
                raise ValueError()
    
    def __call__(self, predict, label):
        predict = predict.cpu()
        label = label.cpu()
        res_dict = dict()
        for metric in self._metrics:
            metric = metric.lower()
            if metric == 'precision':
                res_dict[metric] = precision_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'recall':
                res_dict[metric] = recall_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'accuracy':
                res_dict[metric] = accuracy_score(y_true=label, y_pred=predict)
            elif metric == 'f1_score':
                res_dict['micro_f1'] = f1_score(y_true=label, y_pred=predict, average='micro')
                res_dict['macro_f1'] = f1_score(y_true=label, y_pred=predict, average='macro')
            elif metric == 'micro_f1':
                res_dict['micro_f1'] = f1_score(y_true=label, y_pred=predict, average='micro')
            elif metric == 'macro_f1':
                res_dict['macro_f1'] = f1_score(y_true=label, y_pred=predict, average='macro')
        return res_dict