#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/23 3:35
# @Author : ZM7
# @File : load_data
# @Software: PyCharm
import dgl
import numpy as np
import rgcn.utils as r_utils

def load_data(data_name='ICEWS14s'):
    data = r_utils.load_data(data_name)
    num_nodes, num_rels, train_list, valid_list, test_list = load_data_list(data_name)
    total_data = train_list + valid_list + test_list
    time_num = [len(np.unique(da[:, [0, 2]])) for da in total_data]  # 每个时刻出现的实体数量
    total_times = range(len(total_data))
    time_idx = np.zeros(len(time_num) + 1, dtype=int)
    time_idx[1:] = np.cumsum(time_num)  # 不同时刻的编号
    # 计算不同数据集的起始 ID
    train_sid = 0
    valid_sid = len(train_list)
    test_sid = len(valid_list) + valid_sid
    all_ans_list_test = r_utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = r_utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = r_utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = r_utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)
    graph, data_dic = dgl.load_graphs('./graph_' + data_name)
    graph = graph[0]
    node_id_new = data_dic['s_index']
    s_t = data_dic['s_t']
    s_f = data_dic['s_f']
    s_l = data_dic['s_l']
    return num_nodes, num_rels, train_list, valid_list, test_list, total_data, all_ans_list_test, all_ans_list_r_test,\
           all_ans_list_valid, all_ans_list_r_valid,\
           graph, node_id_new, s_t, s_f, s_l, train_sid, valid_sid, test_sid, total_times, time_idx


def load_data_list(data_name):
    data = r_utils.load_data(data_name)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    train_list = r_utils.split_by_time(data.train)
    valid_list = r_utils.split_by_time(data.valid)
    test_list = r_utils.split_by_time(data.test)
    return num_nodes, num_rels, train_list, valid_list, test_list


