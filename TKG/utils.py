#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/7/6 9:07
# @Author : ZM7
# @File : utils
# @Software: PyCharm


import numpy as np
import os
import dgl
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from dgl.sampling import sample_neighbors



import sys
class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2], quad[3]] for quad in data if quad[3] == tim]
    return np.array(triples)


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            if len(line_split) < 4:
                continue
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    return np.array(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def cal_length(triple, s_f, s_t, t, L=365, max_length=20, data_length=400, name=None):
    """

    :param triple: 当前的triples
    :param s_f: 当前节点的历史编号
    :param s_t:
    :param t:
    :param L:
    :param max_length:
    :return:
    """
    if name == 'ICEWS05-15' or name == 'GDELT':
        if t-data_length < 0:
            s_f = s_f[:,0:t+1]
            s_t = s_t[:, 0:t + 1]
        else:
            s_f = s_f[:, t-data_length:t + 1]
            s_t = s_t[:, t - data_length:t + 1]

    entity, idx = np.unique(triple[:, 0], return_inverse=True)
    s_f = s_f[entity, 0:t+1]    # 0到t时刻s的索引
    s_t = s_t[entity, 0:t+1]    # 0到t时刻s发生交互的时间
    en_l = np.zeros((len(entity), max_length), dtype=int) # 存实体
    t_l = L*np.ones((len(entity), max_length), dtype=int)  # 存实体发生时间
    entity_set = []
    time_set = []
    length = np.zeros(len(entity), dtype=int)
    for i in range(len(entity)):
        all_time = np.unique(s_t[i])[0:-1]
        if len(all_time) == 0:
            continue
        else:
            all_entity = np.unique(s_f[i])[0:-1]
            if len(all_entity) < max_length:
                en_l[i][0:len(all_entity)] = all_entity
                t_l[i][0:len(all_entity)] = all_time
                length[i] = len(all_entity)
                entity_set.append(all_entity)
                time_set.append(all_time)
            else:
                en_l[i] = all_entity[-max_length:]
                t_l[i] = all_time[-max_length:]
                length[i] = max_length
                entity_set.append(all_entity[-max_length:])
                time_set.append(all_time[-max_length:])
    if len(entity_set) == 0:
        entity_set.append([0])
        time_set.append([0])
    return torch.from_numpy(en_l), torch.from_numpy(t_l), torch.from_numpy(np.unique(np.concatenate(entity_set))),\
           np.unique(np.concatenate(time_set)), torch.from_numpy(length)


def original_order(ordered, indices):
    return ordered.ordered.gather(1, indices.argsort(1))


class decoder_sorce(nn.Module):
    def __init__(self, in_dim, score='mlp'):
        super(decoder_sorce, self).__init__()
        self.h_dim = in_dim
        self.score = score
        if self.score == 'mlp':
            self.linear_1 = nn.Linear(self.h_dim*2, self.h_dim, bias=False)

    def forward(self, head_embedding, rel_embedding, tail_embedding, triple):
        if self.score == 'mlp':
            h_embedding = head_embedding[triple[:,0]]
            r_embedding = rel_embedding[triple[:,1]]
            x = self.linear_1(torch.cat((h_embedding, r_embedding), 1))
            x = F.relu(x)
            s = torch.mm(x, tail_embedding.transpose(1, 0))
        return s


def loader(total_data, max_batch, start_id, no_batch=False, mode='train'):
    e_num_time = [len(da) for da in total_data] # 每个时刻三元组的数量
    all_data = []
    all_time = []
    for t, data in enumerate(total_data):
        if mode == 'train':
            if t ==0:
                continue
        if no_batch:
            all_data.append(data)
            all_time.append(start_id+t)
        else:
            g_num = (len(data)//max_batch) + 1  #划分的组数
            indices = np.random.permutation(len(data))
            for i in range(g_num):
                if len(indices[i*max_batch:(i+1)*max_batch]) == 0:
                    continue
                all_data.append(data[indices[i*max_batch:(i+1)*max_batch]])
                all_time.append(start_id+t)
    return all_data, all_time


class myFloder(Dataset):
    def __init__(self, total_data, max_batch=100, start_id=0, no_batch=False, mode='train'):
        self.data = loader(total_data, max_batch, start_id, no_batch, mode)
        self.size = len(self.data[0])


    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return self.size



class Collate():
    def __init__(self, num_nodes, num_rels, s_f, s_t, total_length, name='ICEWS14s', encoder='rgat', decoder='rgat',
                 max_length=10, all=True, graph=None, k=2):
        self.encoder = encoder
        self.decoder = decoder
        self.g = graph
        self.k = k
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.s_f = s_f
        self.s_t = s_t
        self.max_length = max_length
        self.total_length = total_length
        self.name = name
        self.all = all

    def collate_rel(self, data, data_length=400):
        triple, t = data[0]
        if self.name in ['GDELT', 'ICEWS05-15']:
            e_dix = (self.g.edata['e_s'] < t) & (self.g.edata['e_s_'] > t-data_length)
            graph = dgl.edge_subgraph(self.g, e_dix, relabel_nodes=False)
        else:
            graph = self.g
        # 元组扩增
        if triple.shape[-1] == 3:
            inverse_triple = triple[:, [2, 1, 0]]
        else:
            inverse_triple = triple[:, [2, 1, 0, 3]]
        inverse_triple[:, 1] = triple[:, 1] + self.num_rels
        triple = np.vstack([triple, inverse_triple])
        data_list = {}
        # 计算采样所需要的东西
        sample_list, time_list, sample_unique, time_unique, list_length = cal_length(triple, self.s_f, self.s_t, t,
                                                                                     self.total_length,
                                                                                     self.max_length,
                                                                                     data_length=data_length,
                                                                                     name=self.name)
        if time_unique[-1] == t:
            time_unique = time_unique[0:-1]
        data_list['triple'] = torch.tensor(triple)
        if self.encoder in ['rgat', 'regcn'] or self.decoder in ['rgat','rgcn','rgat_r','rgat_r1']:
            # 先采样3阶邻居子图：
            sub_node = sample_k_neighbor(graph, sample_unique, self.k)
            sub_graph = dgl.node_subgraph(graph, sub_node)  #k阶子图
            old_n_id = sub_graph.ndata[dgl.NID]
            # 为了进一步缩小规模，再根据时间采样子图, sub_d_graph
            sub_d_eid = np.in1d(sub_graph.edata['e_t'], time_unique) | \
                        (np.in1d(sub_graph.edata['e_rel_o'], sample_unique)
                         & np.in1d(sub_graph.edata['e_rel_h'], sample_unique))
            sub_d_graph = dgl.edge_subgraph(sub_graph, torch.from_numpy(sub_d_eid))
        # 采样出sub_d_graph用于decoder
        if self.decoder in ['rgat', 'rgcn','rgat_r','rgat_x','regcn','rgat1','rgat_r1']:
            data_list['sub_d_graph'] = sub_d_graph
            data_list['pre_d_nid'] = old_n_id[sub_d_graph.ndata[dgl.NID]]  # sub_d_graph的原始节点编号
        # 再根据e_t采样出sub_e_graph用于encoder
        if self.encoder in ['rgat', 'rgcn','regcn','rgat_r1']:
            sub_e_id = np.in1d(sub_d_graph.edata['e_t'], time_unique)
            sub_e_graph = dgl.edge_subgraph(sub_d_graph, torch.from_numpy(sub_e_id))
            data_list['sub_e_graph'] = sub_e_graph
            data_list['pre_e_eid'] = sub_d_graph.edata[dgl.NID][sub_e_graph.edata[dgl.NID]]  # 三阶子图边的标号
            data_list['pre_e_nid'] = old_n_id[sub_d_graph.ndata[dgl.NID][sub_e_graph.ndata[dgl.NID]]]  # 原始图节点的标号

        data_list['sample_list'] = sample_list
        data_list['time_list'] = time_list
        data_list['list_length'] = list_length
        data_list['t'] = torch.tensor([t])
        data_list['sample_unique'] = sample_unique
        data_list['time_unique'] = torch.LongTensor(time_unique)
        return data_list


def sample_k_neighbor(g, seed_nodes, k):
    temp = seed_nodes
    all_nodes = seed_nodes
    for _ in range(k):
        in_nodes =  np.unique(torch.cat(sample_neighbors(g, temp, fanout=-1, edge_dir='in').edges()))
        out_nodes = np.unique(torch.cat(sample_neighbors(g, temp, fanout=-1, edge_dir='out').edges()))
        temp = np.setdiff1d(np.unique((in_nodes, out_nodes)), all_nodes)
        all_nodes = np.unique(np.concatenate((all_nodes, temp)))
    return all_nodes


def mkdir(path):
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return
    else:
        return

def mkdir_if_not_exist(file_name):
    import os
    import shutil

    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def upto_k_neighbor_nodes(g, seed_nodes, k):
    for _ in range(k):
        in_nodes = list(torch.cat(dgl.sampling.sample_neighbors(g, seed_nodes, fanout=-1, edge_dir='in').edges()).numpy())
        out_nodes = list(torch.cat(dgl.sampling.sample_neighbors(g, seed_nodes, fanout=-1, edge_dir='out').edges()).numpy())
        new_nodes = set(in_nodes + out_nodes)
        seed_nodes = list(new_nodes | set(seed_nodes))
    return seed_nodes


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm
