# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
import pandas as pd

import torch
import torch_geometric.nn
from torch_geometric.data import HeteroData
import torch.nn as nn

def read_file(path: str):
    """
    读取文件并且清理不需要的键
    :param path: 文件路径
    :return: 清理过的数据
    """
    with open(path, "rb") as f:
        data = pkl.load(f)
    """
    for i in range(len(data)):
        print(len(data[i]['lans']))
    """
    clean_data = []
    #clean_data = data[-5]
    for chart in data:
        del chart['proj'], chart['methods'], chart['method_hugecode'],  chart['method_lines'], chart['rtest_methods'],  chart['ftest_methods'],  chart['lcorrectnum']
        clean_data.append(chart)
    return clean_data

def deal_data(raw_data: dict):
    """
    将数据处理成异构图
    :param raw_data: 传入的字典参数
    :return: graph: 构建好的异构图
    """
    #构建图特征, 几种ltype作为特征embedding，rtest, ftest用0,1表示，其他为2——8构建embedding字典
    graph = HeteroData() #创建异构图
    cfg_lines = raw_data['cfg_lines']
    cfg_lines = torch.from_numpy(np.array(cfg_lines).T).int()
    graph['lines', 'cfg', 'lines'].edge_index = cfg_lines #将CFG边放入异构图

    ast_lines = raw_data['lines_lines']
    ast_lines = torch.from_numpy(np.array(ast_lines).T).int()
    graph['lines', 'ast', 'lines'].edge_index = ast_lines  # 将AST边放入异构图

    dfg_lines = raw_data['dfg_lines']
    dfg_lines = torch.from_numpy(np.array(dfg_lines).T).int()
    graph['lines', 'dfg', 'lines'].edge_index = dfg_lines  # 将DFG边放入异构图

    rtest_lines = raw_data['rtest_lines']
    rtest_lines = torch.from_numpy(np.array(rtest_lines).T).int()
    graph['rtest', 'have', 'lines'].edge_index = rtest_lines  # 将rtest-line从属关系放入异构图

    ftest_lines = raw_data['ftest_lines']
    ftest_lines = torch.from_numpy(np.array(ftest_lines).T).int()
    graph['ftest', 'have', 'lines'].edge_index = ftest_lines  # 将ftest-line从属关系放入异构图

    lans = raw_data['lans']
    num_lines = len(raw_data['lines'])
    labels = np.zeros(num_lines)
    labels[lans] = 1
    graph['lines'].y = labels

    #生成用于训练，验证和测试的结点
    tol_ids = np.array(list(range(num_lines)))
    np.random.shuffle(tol_ids)
    train_ids = tol_ids[:int(num_lines*0.1)]
    val_ids = tol_ids[int(num_lines*0.1):int(num_lines*0.4)]
    test_ids = tol_ids[int(num_lines*0.4):]
    train_mask = torch.zeros(num_lines)
    train_mask[train_ids] = 1
    val_mask = torch.zeros(num_lines)
    val_mask[val_ids] = 1
    test_mask = torch.zeros(num_lines)
    test_mask[test_ids] = 1
    graph['lines'].train_mask = train_mask.bool()
    graph['lines'].val_mask = val_mask.bool()
    graph['lines'].test_mask = test_mask.bool()
    #print(train_mask)
    #特征构建
    line_types = raw_data['ltype']
    vals = []
    #print(type(line_types))
    for k in line_types.keys():
        val = line_types[k]
        if 'Break' in val:
            vals.append(2)
        elif 'For' in val:
            vals.append(3)
        elif 'If'  in val:
            vals.append(4)
        elif 'Local' in val:
            vals.append(5)
        elif 'Normal' in val:
            vals.append(6)
        elif 'Switch' in val:
            vals.append(7)
        else:
            vals.append(8)

    features = torch.from_numpy(np.array(vals)).long()
    graph['lines'].x = features

    #构建rtest的特征

    rtest_lines = rtest_lines.detach().cpu().numpy().T.tolist()
    features = features.detach().cpu().numpy().tolist()
    num_rtest = len(raw_data['rtest'])
    vals_rtest = [[0] for _ in range(num_rtest)]
    for item in rtest_lines:
        src = item[0]
        dst = item[1]
        vals_rtest[src].append(features[dst])

    for i in range(len(vals_rtest)):
        if len(vals_rtest[i]) > 1:
            vals_rtest[i] = int(sum(vals_rtest[i])/len(vals_rtest[i])+0.5)
        else:
            vals_rtest[i] = 0

    rtest_feature = torch.from_numpy(np.array(vals_rtest)).long()
    graph['rtest'].x = rtest_feature


    ftest_lines = ftest_lines.detach().cpu().numpy().T.tolist()
    num_ftest = len(raw_data['ftest'])
    vals_ftest = [[1] for _ in range(num_ftest)]
    for item in ftest_lines:
        src = item[0]
        dst = item[1]
        vals_ftest[src].append(features[dst])

    for i in range(len(vals_ftest)):
        if len(vals_ftest[i]) > 1:
            vals_ftest[i] = int(sum(vals_ftest[i]) / len(vals_ftest[i])+0.5)
        else:
            vals_ftest[i] = 1

    ftest_feature = torch.from_numpy(np.array(vals_ftest)).long()
    graph['ftest'].x = ftest_feature
    #print(graph)
    return graph

def construct_graph(path: str) -> list:
    """

    :param path:
    :return: graphs_dealed,一个列表，包含所有处理过的图
    """
    data = read_file(path)
    graphs_dealed = []
    for clean_chart in data:
        hetero_graph = deal_data(clean_chart)
        graphs_dealed.append(hetero_graph)
    return graphs_dealed

if __name__ == "__main__":
    path = "data/chart.pkl"
    g_list = construct_graph(path)