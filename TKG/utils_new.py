#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/8/10 8:15
# @Author : ZM7
# @File : utils_new
# @Software: PyCharm

import os
from torch.utils.data import Dataset, DataLoader

def load_data(data_path):
    data_dir = []
    dir_list = os.listdir(data_path)
    dir_list.sort()
    for filename in dir_list:
        data_dir.append(os.path.join(data_path, filename))
    return data_dir


class myFloder_new(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = load_data(root_dir)
        self.size = len(self.dir_list)

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def collate_new(data, encoder='rgat', decoder='rgat'):
    data_list = {}
    data = data[0]
    data_list['sub_e_graph'] = data[0][0]
    data_list['sub_d_graph'] = data[0][1]
    data_list['pre_e_nid'] = data[1]['pre_e_nid']
    data_list['pre_d_nid'] = data[1]['pre_d_nid']
    data_list['t'] = data[1]['t']
    data_list['triple'] = data[1]['triple']
    data_list['sample_list'] = data[1]['sample_list']
    data_list['time_list'] = data[1]['time_list']
    data_list['list_length'] = data[1]['list_length']
    data_list['t'] = data[1]['t']
    data_list['sample_unique'] = data[1]['sample_unique']
    data_list['time_unique'] = data[1]['time_unique']
    return data_list


