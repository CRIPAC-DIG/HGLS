#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/7/5 3:19
# @Author : ZM7
# @File : generate_data
# @Software: PyCharm

import dgl
import os
import numpy as np
import torch
from TKG.utils import get_data_with_t
import argparse
from dgl import save_graphs
from collections import  defaultdict
from TKG.load_data import  load_data_list


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def generate_graph(data, time, time_num, nodes_num=None, rel_nums=None, name='Sample'):
    """
    :param data:
    :param time:
    :param time_num: 每个时刻下entity的数量
    :param nodes_num:
    :param rel_nums:
    :param name:
    :return:
    """
    u = []
    v = []
    rel = []
    rel_1 = []
    u_id = []  # 记录节点的原始编号
    t_id = []  # 记录节点每次出现的时间
    rel_r = []  #时间差
    rel_t = []
    rel_s = []
    rel_s_ = []
    rel_h = []
    rel_o = []
    L = len(time) # 总的时间长度
    time_num = np.roll(time_num, 1)
    time_num[0] = 0
    idx = np.cumsum(time_num)
    num_t = np.zeros(L)
    for i, tim in enumerate(time):
        print(i)
        if name == 'Sample':
            data_1 = get_data_with_t(data, tim)
        else:
            data_1 = data[i]
        src1, rel1, dst1 = data_1[:,[0,1,2]].transpose()
        uniq_v, edges = np.unique((src1, dst1), return_inverse=True)
        n_src1, n_dst1 = np.reshape(edges, (2, -1)) + idx[i]  # 给节点在新图中重新编码
        u.append(np.concatenate((n_src1, n_dst1)))
        u_id.append(uniq_v)
        t_id.append(i*np.ones(len(uniq_v), dtype=int))
        v.append(np.concatenate((n_dst1, n_src1)))
        rel.append(np.concatenate((rel1, rel1+rel_nums)))
        rel_1.append(np.concatenate((rel1, rel1+rel_nums)))
        rel_r.append(0 * np.ones(len(rel1) * 2, dtype=int))
        rel_t.append(i*np.ones(len(rel1)*2, dtype=int))
        rel_s.append(i*np.ones(len(rel1)*2, dtype=int))
        rel_s_.append(i*np.ones(len(rel1)*2, dtype=int))
        rel_h.append(np.concatenate((n_src1, n_dst1)))
        rel_o.append(np.concatenate((n_dst1, n_src1)))

        if i == len(time)-1:
            break
        for j, tim in enumerate(time[i+1:]):
            if name == 'Sample':
                data_2 = get_data_with_t(data, time[i+j+1])
            else:
                data_2 = data[i+j+1]
            src2, rel2, dst2 = data_2[:,[0,1,2]].transpose()
            uniq_v2, edges2 = np.unique((src2, dst2), return_inverse=True)
            un_entity = np.intersect1d(uniq_v, uniq_v2)  # 两个时刻子图重复的节点
            if len(un_entity) == 0:
                print('hhh', i, tim)
                continue
            u1 = np.where(np.in1d(uniq_v, un_entity))[0]    # 判断公共节点在不同子图内的索引
            u2 = np.where(np.in1d(uniq_v2, un_entity))[0]   # 判断公共节点在不同子图内的索引
            u.append(np.concatenate((u1 + idx[i], u2 + idx[i+j+1])))
            v.append(np.concatenate((u2 + idx[i+j+1], u1 + idx[i])))
            rel.append(2*rel_nums*np.ones(2*len(un_entity), dtype=int))               # 时间类型的边
            rel_1.append((2 * rel_nums + j) * np.ones(2 * len(un_entity), dtype=int)) # 给不同时间类型的边定义大小
            #rel.append((2 * rel_nums + j) * np.ones(2 * len(un_entity), dtype=int))  # 时间差
            rel_r.append((j+1) * np.ones(2 * len(un_entity), dtype=int))               # 时间差
            rel_t.append(L*np.ones(2*len(un_entity), dtype=int))
            rel_s.append((i+j+1)*np.ones(2*len(un_entity), dtype=int))               #  头实体发生时间
            rel_s_.append(i*np.ones(2*len(un_entity), dtype=int))                    #  尾实体发生时间
            rel_h.append(np.concatenate((u1 + idx[i], u2 + idx[i+j+1])))
            rel_o.append(np.concatenate((u2 + idx[i+j+1], u1 + idx[i])))
    u = np.concatenate(u)
    u_id = np.concatenate(u_id)
    t_id = np.concatenate(t_id)
    v = np.concatenate(v)
    rel = np.concatenate(rel)
    rel_1 = np.concatenate(rel_1)
    rel_r = np.concatenate(rel_r)
    rel_t = np.concatenate(rel_t)
    rel_s = np.concatenate(rel_s)
    rel_s_ = np.concatenate(rel_s_)
    rel_h = np.concatenate(rel_h)
    rel_o = np.concatenate(rel_o)
    graph = dgl.graph((u, v))
    graph.edata['etype'] = torch.LongTensor(rel) # edge relation
    graph.edata['etype1'] = torch.LongTensor(rel_1)
    graph.edata['e_r'] = torch.LongTensor(rel_r)  # 相对时间
    graph.edata['e_t'] = torch.LongTensor(rel_t) # edge time (T is L)
    graph.edata['e_s'] = torch.LongTensor(rel_s) # edge time (头实体发生的时间)
    graph.edata['e_s_'] = torch.LongTensor(rel_s_) # edge time (尾实体发生的时间)
    graph.edata['e_rel_h'] = torch.LongTensor(rel_h)
    graph.edata['e_rel_o'] = torch.LongTensor(rel_o)
    graph.ndata['id'] = torch.from_numpy(u_id).long()
    graph.ndata['t_id'] = torch.from_numpy(t_id).long()
    # ----计算每个时刻的s前一时刻在大图中的位置
    s_his = defaultdict(int)    #记录前一次的索引
    s_his_t = defaultdict(int)  #记录前一次发生的时间
    s_his_f = defaultdict(int)  #记录前一次的索引
    s_his_l = defaultdict(int)  #记录历史序列的长度
    s_last_index = np.zeros((nodes_num, L), dtype=int)   #记录每个时刻的索引
    s_last_t = L*np.ones((nodes_num, L), dtype=int)  # 记录每个时刻上次交互的时间,上次没有发生记为L
    s_last_f = graph.num_nodes() * np.ones((nodes_num, L), dtype=int) #记录每个时刻的索引，没有发生过的节点值为nodes_num
    s_last_l = np.zeros((nodes_num, L), dtype=int)
    node_id = graph.ndata['id'].numpy()
    time_id = graph.ndata['t_id'].numpy()
    # node_id_new = len(node_id) - 1 - np.unique(node_id[::-1], return_index=True)[1]
    # 初始化
    id, node_id_f = np.unique(node_id, return_index=True)
    for n_i, ei in enumerate(id):
        s_his[ei] = node_id_f[n_i]
        s_his_t[ei] = L                                # T+1时间
        s_his_f[ei] = graph.num_nodes()
        s_his_l[ei] = 0                # 0
    s_last_index[id, 0] = node_id_f
    for i, tim in enumerate(time):
        if name == 'Sample':
            data_1 = get_data_with_t(data, tim)
        else:
            data_1 = data[i]
        src1, rel1, dst1 = data_1[:,[0,1,2]].transpose()
        # -----找一个数组记录每个元素这一时刻的索引------
        en = np.unique((src1, dst1))
        if i > 0:
            s_last_index[:, i] = s_last_index[:, i-1]
            s_last_t[:, i] = s_last_t[:, i-1]
            s_last_f[:, i] = s_last_f[:, i-1]
            s_last_l[:, i] = s_last_l[:, i-1]
        for e_i, e in enumerate(en):
            s_last_index[e, i] = s_his[e]
            s_last_t[e, i] = s_his_t[e]
            s_last_f[e, i] = s_his_f[e]
            s_last_l[e, i] = s_his_l[e]
            s_his[e] = e_i + idx[i]        # 计算s上一次交互的编号
            s_his_f[e] = e_i + idx[i]      # 计算s上一次交互的编号
            s_his_t[e] = i                 # 计算s上一次发生交互的时间
            s_his_l[e] = s_his_l[e] + 1    # 计算s历史序列的长度
        # -----------------------------------------
    return graph, torch.from_numpy(s_last_index).long(), torch.from_numpy(s_last_t).long(), \
           torch.from_numpy(s_last_f).long(), torch.from_numpy(s_last_l).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data')
    parser.add_argument('--data', default='Sample')
    args = parser.parse_args()
    print(args)
    num_nodes, num_rels, train_list, valid_list, test_list = load_data_list(args.data, args.space)
    total_data = train_list + valid_list + test_list
    time_num = [len(np.unique(da[:, [0, 2]])) for da in total_data]
    total_times = range(len(total_data))
    save_path = './' + 'graph_' + args.data
    graph, s_index, s_t, s_f, s_l = generate_graph(total_data, total_times, time_num=time_num, nodes_num=num_nodes,
                                                   rel_nums=num_rels, name=args.data)
    save_graphs(save_path, graph, {'s_index': s_index, 's_t': s_t, 's_f': s_f, 's_l': s_l})