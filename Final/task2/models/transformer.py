import os
import sys
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import torchvision
from torchvision import transforms, datasets

from utils import *

class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size, attention_dropout_rate, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = num_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1) 
    
    def transpose_for_scores(self, x):
        process_x_shape = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*process_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        weights = attention_probs if self.vis else None

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        process_context_layer_shape = context_layer.size()[: -2] + (self.all_head_size, )
        context_layer = context_layer.view(*process_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

def numpy_to_torch(weights, trans=False):
    if trans: 
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish_act(x):
    return x * torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate, num_heads, attention_dropout_rate, vis):
        super(Block, self).__init__()
        self.vis = vis 
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = MLP(
            hidden_size=hidden_size, 
            mlp_dim=mlp_dim, 
            dropout_rate=dropout_rate
        )
        self.attn = Attention(
            num_heads=num_heads,
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
            vis=vis
        )
    
    def forward(self, x):
        x_copy = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + x_copy

        x_copy = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + x_copy
        return x, weights
    
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = numpy_to_torch(weights[os.path.join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = numpy_to_torch(weights[os.path.join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = numpy_to_torch(weights[os.path.join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = numpy_to_torch(weights[os.path.join(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(numpy_to_torch(weights[os.path.join(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(numpy_to_torch(weights[os.path.join(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(numpy_to_torch(weights[os.path.join(ROOT, MLP_NORM, "bias")]))

class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate, vis):
        super(Encoder, self).__init__() 
        self.vis = vis
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(
                hidden_size=hidden_size,
                mlp_dim=mlp_dim, 
                dropout_rate=dropout_rate,
                num_heads=num_heads, 
                attention_dropout_rate=attention_dropout_rate,
                vis=vis
            )
            self.layers.append(copy.deepcopy(layer))
        
    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layers:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Embeddings(nn.Module):
    def __init__(self, img_size, patches, hidden_size, dropout_rate, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)
        self.hybrid = None
        patch_size = _pair(patches)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_size,
            kernel_size=patch_size, 
            stride=patch_size
        )

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        batch_num = x.shape[0]
        cls_token = self.cls_token.expand(batch_num, -1, -1)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_token, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings) 
        return embeddings 
        
class Transformer(nn.Module):
    def __init__(self, img_size, patches, hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            img_size=img_size, 
            patches=patches, 
            hidden_size=hidden_size, 
            dropout_rate=dropout_rate
        )
        self.encoder = Encoder(
            hidden_size=hidden_size, 
            num_layers=num_layers,
            mlp_dim=mlp_dim, 
            dropout_rate=dropout_rate,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            vis=vis
        )
    
    def forward(self, input_idx):
        embedding_output = self.embeddings(input_idx)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights

class VIT(nn.Module):
    def __init__(self, classifier, num_output, img_size, patches, hidden_size, num_layers, mlp_dim, dropout_rate, num_heads, attention_dropout_rate, zero_head=False, vis=False):
        super(VIT, self).__init__()
        self.num_output = num_output
        self.zero_head = zero_head
        self.vis = vis
        self.classifier = classifier
        self.img_size = img_size
        self.transformer = Transformer(
            img_size=img_size, 
            patches=patches,
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            mlp_dim=mlp_dim, 
            dropout_rate=dropout_rate, 
            num_heads=num_heads, 
            attention_dropout_rate=attention_dropout_rate,
            vis=vis
        )
        self.head = nn.Linear(hidden_size, num_output)
    
    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights
    
    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(numpy_to_torch(weights["head/kernel"]).t())
                self.head.bias.copy_(numpy_to_torch(weights["head/bias"]).t())
            
            self.transformer.embeddings.patch_embeddings.weight.copy_(numpy_to_torch(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(numpy_to_torch(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(numpy_to_torch(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(numpy_to_torch(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(numpy_to_torch(weights["Transformer/encoder_norm/bias"]))

            pos_emb = numpy_to_torch(weights["Transformer/posembed_input_pos_embedding"])
            process_pos_emb = self.transformer.embeddings.position_embeddings

            if pos_emb.size() == process_pos_emb.size():
                self.transformer.embeddings.position_embeddings.copy_(pos_emb)
            else:
                ntok_new = process_pos_emb.size(1)

                if self.classifier == "token":
                    pos_emb_token, pos_emb_grid = pos_emb[:, :1], pos_emb[0, 1:]
                    ntok_new -= 1
                else:
                    pos_emb_token, pos_emb_grid = pos_emb[:, :0], pos_emb[0]
                
                gs_old = int(np.sqrt(len(pos_emb_grid)))
                gs_new = int(np.sqrt(ntok_new))

                pos_emb_grid = pos_emb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                pos_emb_grid = ndimage.zoom(pos_emb_grid, zoom, order=1)
                pos_emb_grid = pos_emb_grid.reshape(1, gs_new * gs_new, -1)
                pos_emb = np.concatenate([pos_emb_token, pos_emb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(numpy_to_torch(pos_emb))
            
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(numpy_to_torch(weights["conv-root/kernel"], conv=True))
                gn_weight = numpy_to_torch(weights["gn_root/scale"]).view(-1)
                gn_bias = numpy_to_torch(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn_weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn_bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)