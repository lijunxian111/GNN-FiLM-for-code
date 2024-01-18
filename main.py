# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model import GNNFilm
from read_data import construct_graph
from tqdm import tqdm
import numpy as np
import pickle as pkl
import pandas as pd

def eval_node_classifier(model, graph, labels, mask):
    g = graph.to_homogeneous()
    model.eval()
    test_shape = mask.shape[0]
    logits = model(g.x, g.edge_index, g.edge_type)
    pred = logits.argmax(dim=1)
    correct = (pred[:test_shape][mask] == labels[mask]).sum()
    #print(correct)
    acc = int(correct) / (int(mask.sum()) + 1e-6)

    return acc, pred[:test_shape], nn.Sigmoid()(logits)[:, 1].detach().cpu().numpy()

def train_node_classifier_one_graph(model, graph, labels, optimizer, criterion, train_mask):
    """
    训练一个图上的结点分类
    :param model: 传入的模型
    :param graph: 传入的一个异构图
    :param labels: 正确或错误语句的标签
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param train_mask: 用于训练的结点
    :return: 模型和优化器
    """
    g = graph.to_homogeneous()
    g.y = labels

    train_shape = train_mask.shape[0]
    out = model(g.x, g.edge_index, g.edge_type)
    loss = criterion(out[:train_shape][train_mask], g.y[train_mask])
    #loss = criterion(out[:train_shape], g.y)
    if loss.item() > 0:
        return model, loss.item(), loss, optimizer

    return model, None, None, optimizer

def train_node_classifier(model, graph_list, idx,  n_epochs = 200):
    """
    整个训练的函数
    :param model: 传入的模型
    :param graph_list: 异构图列表，每个是一个chart
    :param idx: 用哪个图当测试集, 为整数，是它在列表中的位置
    :param n_epochs: 训练n轮
    :return:
    """
    lenth = len(graph_list)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(n_epochs)):  #每一轮训练
        model.train()
        #rand_num = random.randint(0, lenth-1)
        total_loss = 0.
        total_loss_bk = None
        optimizer.zero_grad()
        for i in range(lenth):  #每一轮把每个图都过一遍
            if i != idx:
                graph = graph_list[i]
                labels = torch.from_numpy(graph['lines'].y).long()
                #mask = graph['lines'].test_mask
                train_mask = graph['lines'].train_mask
                model, one_g_loss, loss_bk, optimizer = train_node_classifier_one_graph(model, graph, labels, optimizer, criterion, train_mask)
                if one_g_loss is not None:
                    total_loss += one_g_loss
                    if total_loss_bk is not None:
                        total_loss_bk = total_loss_bk + loss_bk.clone()
                    else:
                        total_loss_bk = loss_bk.clone()

        total_loss_bk.backward()
        optimizer.step()

        val_labels = torch.from_numpy(graph_list[idx]['lines'].y).long()
        val_mask = graph_list[idx]['lines'].val_mask
        acc, pred, _ = eval_node_classifier(model, graph_list[idx], val_labels, val_mask)
        loss = total_loss / (lenth-1)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')
            #print(f'预测有{np.sum(pred.detach().cpu().numpy())}个错误语句')

    return model

def main():
    """
    主函数, 最后错误语句的列表会放在results.csv中
    :return:
    """
    graph_list = construct_graph('data/chart.pkl')
    model = GNNFilm(in_channels=9, hidden_channels=64, out_channels=2, n_layers=2, num_relations=5) #这里in_channels改成了9，embedding方式变了一下

    model = train_node_classifier(model, graph_list, idx=19, n_epochs=150)
    test_acc, pred, scores = eval_node_classifier(model, graph_list[19], torch.from_numpy(graph_list[19]['lines'].y), graph_list[19]['lines'].test_mask)
    print(f'Test Acc: {test_acc:.3f}')
    test_shape = graph_list[19]["lines"].test_mask.shape[0]
    print(f'预测有{np.sum(pred[:test_shape].detach().cpu().numpy())}个错误语句')
    with open('data/chart.pkl', "rb") as f:
        data = pkl.load(f)

    lines_name = data[19]['lines']
    pred_list = pred[:test_shape].detach().cpu().numpy()
    pred_errors = np.where(pred_list == 1)[0].tolist()
    error_lines = []
    error_score = []

    for k in lines_name.keys():
        if lines_name[k] in pred_errors:
            error_lines.append(k)
            error_score.append(lines_name[k])

    if len(error_lines) == 0:
        print("未预测出错误语句")
    else:
        df = pd.DataFrame({'error_lines': error_lines})
        df.to_csv('results.csv', index=False)

    return

if __name__ == "__main__":
    main()
    """
    model = GNNFilm(in_channels=7, hidden_channels=64, out_channels=2, n_layers=2, num_relations=5)
    g = construct_graph('data/chart.pkl')
    #print(g.edge_types)
    g = g.to_homogeneous()
    #edge_type = g.edge_type
    res = model(g.x, g.edge_index, g.edge_type)
    print(res.shape)
    """