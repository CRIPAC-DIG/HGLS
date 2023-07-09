#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : main
# @Software: PyCharm

import torch
import os
from TKG.utils import  myFloder, Collate, Logger, mkdir_if_not_exist
from rgcn import utils
from torch.utils.data import DataLoader
from TKG.load_data import load_data
from hgls import HGLS
import argparse
import yaml
from yaml import SafeLoader
import datetime
import numpy as np
import dgl
import sys
from sys import exit
from TKG.utils_new import myFloder_new, collate_new
import warnings
warnings.filterwarnings('ignore')
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, total_data, test_dataset, all_ans_list_test, node_id_new, s_t, test_sid):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    test_losses = []
    model.eval()
    for test_data_list in test_dataset:
        with torch.no_grad():
            final_score, final_score_r, test_loss = \
                model(total_data, test_data_list, node_id_new[:, test_data_list['t'][0]].to(device),
                      (test_data_list['t'][0] - s_t[:, test_data_list['t'][0]]).to(device), device=device, mode='test')
            mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_data_list['triple'].to(device),
                                                                                    final_score,
                                                                                    all_ans_list_test[
                                                                                        test_data_list['t'][
                                                                                            0] - test_sid],
                                                                                    eval_bz=1000, rel_predict=0)
            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            test_losses.append(test_loss.item())
    mrr_raw, h1_raw, h3_raw, h10_raw = utils.stat_ranks(ranks_raw)
    mrr_filter, h1_f, h3_f, h10_f = utils.stat_ranks(ranks_filter)
    return np.mean(test_losses), [mrr_raw, h1_raw, h3_raw, h10_raw], [mrr_filter, h1_f, h3_f, h10_f]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGLS')
    parser.add_argument("--gpu", default='0', help="gpu")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    parser.add_argument("--task", type=float, default=0.7, help="weight of entity prediction task")
    parser.add_argument("--short", action='store_true', default=False, help="short-term")
    parser.add_argument("--long", action='store_true', default=False, help="long-term")
    parser.add_argument('--gnn', default='regcn')
    parser.add_argument('--fuse', default='con', help='entity fusion')
    parser.add_argument('--r_fuse', default='re', help='relation fusion')
    # parser.add_argument("--r_p", action='store_true', default=True, help="tkg")     # 关系预测
    parser.add_argument("--record", action='store_true', default=False, help="save log file")
    parser.add_argument("--model_record", action='store_true', default=False, help="save model file")
    # configuration for optimal parameters
    parser.add_argument('--config', type=str, default='long_config.yaml')
    args = parser.parse_args().__dict__                                          # REGCN 的参数
    short_con = yaml.load(open('short_config.yaml'), Loader=SafeLoader)[args['dataset']]
    long_con = yaml.load(open('long_config.yaml'), Loader=SafeLoader)[args['dataset']]
    log_file = f'{args["dataset"]}_short_{args["short"]}_long_{args["long"]}_' \
               f'f_{args["fuse"]}_fr_{args["r_fuse"]}_ta_{args["task"]}' \
               f'_gnn1_{long_con["encoder"]}_{long_con["a_layer_num"]}_gnn2_{long_con["decoder"]}_{long_con["d_layer_num"]}' \
               f'_seq_{short_con["sequence"]}_{short_con["sequence_len"]}_max_length_{long_con["max_length"]}_fil_{long_con["filter"]}_ori_{long_con["ori"]}' \
               f'last_{long_con["last"]}'
    if args['record']:
        log_file_path = f'results/g_{args["gpu"]}_' + log_file
        mkdir_if_not_exist(log_file_path)
        sys.stdout = Logger(log_file_path)
        print(f'Logging to {log_file_path}')

    num_nodes, num_rels, train_list, valid_list, test_list, total_data, all_ans_list_test, all_ans_list_r_test, \
    all_ans_list_valid, all_ans_list_r_valid, graph, node_id_new, s_t, s_f, s_l, train_sid, valid_sid, test_sid, \
    total_times, time_idx = load_data(args['dataset'])
    #选择环境
    device = torch.device('cuda:0')
    os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
    # regcn的参数补充
    num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    short_con['num_static_rels'] = num_static_rels
    short_con['num_words'] = num_words
    # HGLS的参数补充
    long_con['time_length'] = len(total_data)
    long_con['time_idx'] = time_idx
    print(args)
    print(short_con)
    print(long_con)

    model = HGLS(graph.to(device), num_nodes, num_rels, args['n_hidden'], args['task'], args['relation_prediction'],
                 args['short'], args['long'], args['fuse'], args['r_fuse'], short_con, long_con).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    model.apply(inplace_relu)
    if args['dataset'] in ['ICEWS05-15', 'ICEWS18', 'GDELT']:
        print('load data from folder')
        train_path = 'data/' + '_' + args['dataset'] + '/train/'
        valid_path = 'data/' + '_' + args['dataset'] + '/val/'
        test_path = 'data/' + '_' + args['dataset'] + '/test/'
        train_set = myFloder_new(train_path, dgl.load_graphs)
        val_set = myFloder_new(valid_path, dgl.load_graphs)
        test_set = myFloder_new(test_path, dgl.load_graphs)
        train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=collate_new, shuffle=True, pin_memory=True, num_workers=8)
        val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True, num_workers=3)
        test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True, num_workers=3)
    else:
        print('load data online')
        train_set = myFloder(train_list, max_batch=100, start_id=train_sid, no_batch=True, mode='train')
        val_set = myFloder(valid_list, max_batch=100, start_id=valid_sid, no_batch=True, mode='test')
        test_set = myFloder(test_list, max_batch=100, start_id=test_sid, no_batch=True, mode='test')
        co = Collate(num_nodes, num_rels, s_f, s_t, len(total_data), args['dataset'], long_con['encoder'], long_con['decoder'], max_length=long_con['max_length'], all=False, graph=graph, k=2)
        train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=co.collate_rel, shuffle=True, pin_memory=True, num_workers=8)
        val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=co.collate_rel, shuffle=False, pin_memory=True, num_workers=4)
        test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=co.collate_rel, shuffle=False, pin_memory=True, num_workers=4)

    for epoch in range(args['n_epochs']):
        print('Epoch {}'.format(epoch), '_', 'Start training: ', datetime.datetime.now(),
              '=============================================')
        model.train()
        stop = True
        losses = [0]
        loss_es = [0]
        loss_rs = []
        for train_data_list in train_dataset:
            loss_e, loss_r, loss = model(total_data, train_data_list, node_id_new[:, train_data_list['t'][0]].to(device),
                                        (train_data_list['t'][0] - s_t[:, train_data_list['t'][0]]).to(device), device=device, mode='train')
            losses.append(loss.item())
            loss_es.append(loss_e.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm'])  # clip gradients
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch {}, loss {:.4f}'.format(epoch, np.mean(losses)), datetime.datetime.now())
        print('\tStart validating: ', datetime.datetime.now())
        val_result = test(model, total_data, val_dataset, all_ans_list_valid, node_id_new, s_t, valid_sid)
        print('\ttrain_loss:%.4f\tval_loss:%.4f\tval_Mrr_raw:%.4f\tval_Hits(raw)@1:%.4f\tval_Hits(raw)@3:%.4f\tval_Hits(raw)@10:%.4f'
              '\tval_Mrr_filter:%.4f\tval_Hits(filter)@1:%.4f\tval_Hits(filter)@3:%.4f\tval_Hits(filter)@10:%.4f' %
              (np.mean(losses), val_result[0], val_result[1][0], val_result[1][1], val_result[1][2], val_result[1][3],
               val_result[2][0], val_result[2][1], val_result[2][2], val_result[2][3]))
        print('\tStart testing: ', datetime.datetime.now())
        test_result = test(model, total_data, test_dataset, all_ans_list_test, node_id_new, s_t, test_sid)
        print('\tval_loss:%.4f\tval_Mrr_raw:%.4f\tval_Hits(raw)@1:%.4f\tval_Hits(raw)@3:%.4f\tval_Hits(raw)@10:%.4f'
              '\tval_Mrr_filter:%.4f\tval_Hits(filter)@1:%.4f\tval_Hits(filter)@3:%.4f\tval_Hits(filter)@10:%.4f' %
              (test_result[0], test_result[1][0], test_result[1][1], test_result[1][2], test_result[1][3],
               test_result[2][0], test_result[2][1], test_result[2][2], test_result[2][3]))






