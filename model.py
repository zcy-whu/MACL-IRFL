import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv

class HGNNLayer(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, n_hid: int, n_heads: int, norm: bool, dropout: float):
        super(HGNNLayer, self).__init__()

        self.n_hid = n_hid
        self.n_heads = n_heads
        self.norm = norm
        self.dropout = dropout

        # intra reltion aggregation modules
        self.intra_rel_agg = nn.ModuleDict({
            etype: GATConv(n_hid, n_hid, n_heads, feat_drop=dropout, allow_zero_in_degree=True)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        # gate mechanism 
        self.res_fc = nn.ModuleDict()
        self.res_weight = nn.ParameterDict()
        for ntype in graph.ntypes:
            self.res_fc[ntype] = nn.Linear(n_hid, n_heads * n_hid)
            self.res_weight[ntype] = nn.Parameter(torch.randn(1))
        self.leakyrelu = nn.LeakyReLU()

        # LayerNorm
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: nn.LayerNorm(n_hid) for ntype in graph.ntypes})

        self.dense_biinter = nn.ModuleDict({ntype: nn.Linear(n_hid, n_hid) for ntype in graph.ntypes})
        self.dense_siinter = nn.ModuleDict({ntype: nn.Linear(n_hid, n_hid) for ntype in graph.ntypes})
        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        """Reinitialize learnable parameters"""
        gain = nn.init.calculate_gain('relu')
        for ntype in self.res_fc:
            nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for ntype in self.dense_biinter:
            init_weights(self.dense_biinter[ntype])
        for ntype in self.dense_siinter:
            init_weights(self.dense_siinter[ntype])


    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)  
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension) 
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = self.leakyrelu(fun_bi(deep_fm))
        bias_fm = self.leakyrelu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm


    def forward(self, graph: dgl.DGLGraph, node_features: dict, relations: list):
        # same type neighbors aggregation
        intra_features = {}
        for stype, etype, dtype in graph.canonical_etypes:
            if etype in relations:
                rel_graph = graph[stype, etype, dtype]
                dst_feat = self.intra_rel_agg[etype](rel_graph, (node_features[stype], node_features[dtype]), rel_graph[(stype, etype, dtype)].edata['w'])
                intra_features[(stype, etype, dtype)] = dst_feat.squeeze()

        # different types aggregation 
        inter_features = {}
        for ntype in graph.ntypes:
            types_features = []
            for stype, etype, dtype in intra_features:
                if ntype == dtype:
                    types_features.append(intra_features[(stype, etype, dtype)])
            types_features = torch.stack(types_features, dim=1)
            out_feat = self.feat_interaction(types_features, self.dense_biinter[ntype], self.dense_siinter[ntype], dimension=1)
            inter_features[ntype] = out_feat


        new_features = {}  
        for ntype in inter_features:
            new_features[ntype] = {}
            alpha = torch.sigmoid(self.res_weight[ntype])
            new_features[ntype] = inter_features[ntype] + node_features[ntype] * alpha
            if self.norm:
                new_features[ntype] = self.norm_layer[ntype](new_features[ntype])  

        return new_features


class HGNN(nn.Module):
    def __init__(self, graph: dgl.DGLGraph, report_num, file_num, train_report_ids, val_report_ids, test_report_ids, n_hid: int, n_layers: int, n_heads: int, norm: bool, dropout: float = 0.2):
        super(HGNN, self).__init__()

        self.n_hid     = n_hid
        self.n_layers  = n_layers
        self.n_heads   = n_heads
        self.node_num = {'report': report_num, 'file': file_num}
        self.report_ids = {'train': train_report_ids, 'val': val_report_ids, 'test': test_report_ids}
        self.loss_fn = nn.CrossEntropyLoss()

        self.gnn_layers = nn.ModuleList([HGNNLayer(graph, n_hid, n_heads, norm, dropout) for _ in range(n_layers)])
        self.embed = nn.ModuleDict({ntype: nn.Embedding(self.node_num[ntype], n_hid) for ntype in graph.ntypes})

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embed.values():
            nn.init.xavier_uniform_(emb.weight)


    def forward(self, graph: dgl.DGLGraph, labels, mode, relations):
        inp_feat = {}
        for ntype in graph.ntypes:
            inp_feat[ntype] = self.embed[ntype](graph.nodes[ntype].data['_ID'])

        # gnn
        for i in range(self.n_layers):
            inp_feat = self.gnn_layers[i](graph, inp_feat, relations)

        scores = torch.matmul(inp_feat['report'], inp_feat['file'].t())
        if mode == 'train':
            pred = scores[:len(self.report_ids['train'])]
        elif mode == 'val':
            pred = scores[len(self.report_ids['train']):(len(self.report_ids['train']) + len(self.report_ids['val']))]
        elif mode == 'test':
            pred = scores[-len(self.report_ids['test']):]

        loss = self.loss_fn(pred, labels)

        return inp_feat, pred, loss
