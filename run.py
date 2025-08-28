import pickle
import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from dgl.data.utils import load_graphs
from model import HGNN
from utils import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
transform = TSNE


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def contrastive_loss(z_i, z_j, temp, batch_size, loss_fn):
    N = 2 * batch_size
    z = torch.cat((z_i, z_j), dim=0)
    sim = torch.mm(z, z.T) / temp
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    cts_loss = loss_fn(logits, labels)
    return cts_loss


def evaluate(model, graph, labels, positive_indices, mode):
    model.eval()
    with torch.no_grad():
        _, scores, loss = model[1](graph, labels, mode, ['similar', 'similar_by', 'co-cited'])
        map, mrr, hit_k = metrics(scores, positive_indices)
        print(f'val map: {map}, val mrr: {mrr}, val hit_k: {hit_k}')

    return map, mrr, hit_k

def parse_args():
    parser = argparse.ArgumentParser(description="Run GNNBL.")
    parser.add_argument('--data', type=str, choices=['birt', 'eclipse', 'jdt', 'swt', 'tomcat'], default='jdt', help="dataset.")
    parser.add_argument('--epochs', type=int, default=800, help="training epoches")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--num_layers', type=int, default=4, help="gnn layers number")
    parser.add_argument('--num_hidden', type=int, default=32, help="hidden layer dimention")
    parser.add_argument('--early_stop', type=int, default=10, help='Whether to stop training early.')
    parser.add_argument("--base_path", default='./processed_data/', help="data url")
    parser.add_argument("--save_path", default='./results/', help="base url")
    parser.add_argument("--model_path", default='checkpoint.pt', help="save model or not")
    parser.add_argument("--just_test", default=True, help="just test the model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # birt, eclipse, jdt, swt, tomcat
    project_name = args.data
    model_out_path = f'{args.save_path}{project_name}/{args.model_path}'

    g, _ = load_graphs(f'{args.base_path}{project_name}/{project_name}.bin')
    with open(f'{args.base_path}{project_name}/labels', 'rb') as f:
        labels_dict = pickle.load(f)
    with open(f'{args.base_path}{project_name}/preprocessed_reports/train_reports', 'rb') as f:
        train_reports = pickle.load(f)
    with open(f'{args.base_path}{project_name}/preprocessed_reports/val_reports', 'rb') as f:
        val_reports = pickle.load(f)
    with open(f'{args.base_path}{project_name}/preprocessed_reports/test_reports', 'rb') as f:
        test_reports = pickle.load(f)

    graph = g[0]
    print(graph)
    train_report_ids = torch.arange(0, len(train_reports))
    train_file_ids = graph.ndata['_ID']['file']
    val_report_ids = torch.arange(len(train_reports), len(train_reports) + len(val_reports))
    test_report_ids = torch.arange(len(train_reports) + len(val_reports), len(train_reports) + len(val_reports) + len(test_reports))

    train_labels, val_labels, test_labels, train_positive_indices, val_positive_indices, test_positive_indices = prepare_labels(labels_dict, len(train_file_ids))
    report_num = graph.number_of_nodes('report')
    file_num = graph.number_of_nodes('file')

    model1 = HGNN(graph=graph, report_num=report_num, file_num=file_num, train_report_ids=train_report_ids,
                 val_report_ids=val_report_ids, test_report_ids=test_report_ids, n_hid=args.num_hidden, n_layers=args.num_layers, n_heads=1, norm=True)
    model2 = HGNN(graph=graph, report_num=report_num, file_num=file_num, train_report_ids=train_report_ids,
                  val_report_ids=val_report_ids, test_report_ids=test_report_ids, n_hid=args.num_hidden,
                  n_layers=args.num_layers, n_heads=1, norm=True)
    model = nn.Sequential(model1, model2)
    print(f'# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  
    loss_fn = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=args.early_stop, verbose=True, path=model_out_path)
    for epoch in range(args.epochs):
        model.train()

        h1, scores1, loss1 = model[0](graph, train_labels, 'train', ['located', 'located_by'])
        h2, scores2, loss2 = model[1](graph, train_labels, 'train', ['similar', 'similar_by', 'co-cited'])

        contrast_loss = (contrastive_loss(h1['report'].data, h2['report'], 0.1, h1['report'].shape[0], loss_fn) +
                         contrastive_loss(h1['file'].data, h2['file'], 0.1, h1['file'].shape[0], loss_fn))
        total_loss = loss1 + loss2 + 0.01 * contrast_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()
        print(f'epoch: {epoch}, loss: {total_loss}')

        map, mrr, hit_k = metrics(scores2, train_positive_indices)  
        print(f'train loss: {total_loss}, train map: {map}, train mrr: {mrr}, train hit_k: {hit_k}')

        val_map, val_mrr, val_hit_k = evaluate(model, graph, val_labels, val_positive_indices, 'val')

        early_stopping(total_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(model_out_path))
    test_map, test_mrr, test_hit_k = evaluate(model, graph, test_labels, test_positive_indices, 'test')

