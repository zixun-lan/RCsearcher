import dgl
import torch
from dgl.nn.pytorch.conv import EGATConv


def collate_RCI(list_of_states):  # product_dgl, product_fp, Placed_RcNodeIdx, RcNodeIdx_list, mask, t
    batch_product_dgl, batch_product_fp, batch_placed_rc, _rc_list, batch_mask, _t = map(list, zip(*list_of_states))
    batch_product_dgl = dgl.batch(batch_product_dgl)
    batch_product_fp = torch.cat(batch_product_fp, dim=0)  # bxd
    batch_placed_rc = torch.cat(batch_placed_rc)  # bn
    batch_mask = torch.cat(batch_mask, dim=0)  # b(n+1)x1
    return batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask  # dgl, bn, bxd, b(n+1)x1


class EGATLayer(torch.nn.Module):
    def __init__(self, hidden_dimension, num_heads):
        super(EGATLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.num_heads = num_heads
        self.egat_layer = EGATConv(in_node_feats=hidden_dimension,
                                   in_edge_feats=hidden_dimension,
                                   out_node_feats=hidden_dimension,
                                   out_edge_feats=hidden_dimension,
                                   num_heads=num_heads)
        self.mlp_node = torch.nn.Linear(hidden_dimension * num_heads, hidden_dimension)
        self.mlp_edge = torch.nn.Linear(hidden_dimension * num_heads, hidden_dimension)

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            new_node_feats, new_edge_feats = self.egat_layer(graph, nfeats, efeats)

            new_node_feats = new_node_feats.reshape(-1, self.hidden_dimension * self.num_heads)
            new_node_feats = self.mlp_node(new_node_feats)
            new_node_feats = torch.nn.functional.elu(new_node_feats)  # nxd

            new_edge_feats = new_edge_feats.reshape(-1, self.hidden_dimension * self.num_heads)
            new_edge_feats = self.mlp_edge(new_edge_feats)
            new_edge_feats = torch.nn.functional.elu(new_edge_feats)  # Exd

        return new_node_feats, new_edge_feats  # nxd, Exd


class EGATModel(torch.nn.Module):
    def __init__(self, hidden_dimension, num_heads, num_layers, residual=True):
        super(EGATModel, self).__init__()
        self.residual = residual
        self.layers = torch.nn.ModuleList([EGATLayer(hidden_dimension=hidden_dimension, num_heads=num_heads)
                                           for _ in range(num_layers)])

    def forward(self, graph, nfeats, efeats):
        with graph.local_scope():
            for layer in self.layers:
                if self.residual:
                    residual_n = nfeats
                    residual_e = efeats
                    nfeats, efeats = layer(graph, nfeats, efeats)
                    nfeats = nfeats + residual_n
                    efeats = efeats + residual_e
                else:
                    nfeats, efeats = layer(graph, nfeats, efeats)
        return nfeats, efeats  # nxd, Exd


class BaseRCIEncoder(torch.nn.Module):
    def __init__(self, hidden_dimension, num_egat_heads, num_egat_layers, residual=True, have_fp=True):
        super(BaseRCIEncoder, self).__init__()
        # ablation
        self.residual = residual
        self.have_fp = have_fp

        # encode FP
        if have_fp:
            self.dense_FP = torch.nn.Linear(2048, hidden_dimension)

        # encode product
        self.dense_init_nfeats = torch.nn.Linear(81, hidden_dimension)
        self.dense_init_efeats = torch.nn.Linear(17, hidden_dimension)
        self.product_graph_encoder = EGATModel(hidden_dimension, num_egat_heads, num_egat_layers, residual=residual)
        self.Placed_rc_embedding = torch.nn.Embedding(2, hidden_dimension)
        if have_fp:
            self.dense_product_graph_level = torch.nn.Linear(3 * hidden_dimension, hidden_dimension)
        else:
            self.dense_product_graph_level = torch.nn.Linear(2 * hidden_dimension, hidden_dimension)

    def forward(self, batch_product_dgl, batch_placed_rc, batch_product_fp):  # batch_dgl, bn, bxd, b(n+1)x1
        with batch_product_dgl.local_scope():
            # encode FP
            if self.have_fp:
                encoded_product_fp = torch.nn.functional.elu(self.dense_FP(batch_product_fp))  # bxd

            # encode product to node-embedding
            init_x = self.dense_init_nfeats(batch_product_dgl.ndata['x'])  # nxd
            init_e = self.dense_init_efeats(batch_product_dgl.edata['e'])  # Exd
            placed_rc = self.Placed_rc_embedding(batch_placed_rc.type(torch.long))  # nxd
            x_add_placed_rc = init_x + placed_rc  # mxd
            node_emb, edge_emb = self.product_graph_encoder(graph=batch_product_dgl, nfeats=x_add_placed_rc,
                                                            efeats=init_e)  # bnxd, bExd

            # product_graph_level_embedding
            batch_product_dgl.ndata['node_emb'] = node_emb
            batch_product_dgl.edata['edge_emb'] = edge_emb
            batch_graph_emb_by_node = dgl.readout_nodes(graph=batch_product_dgl, feat='node_emb', op="mean")  # bxd
            batch_graph_emb_by_edge = dgl.readout_edges(graph=batch_product_dgl, feat='edge_emb', op="mean")  # bxd
            if self.have_fp:
                batch_graph_emb = torch.cat([batch_graph_emb_by_node, batch_graph_emb_by_edge, encoded_product_fp],
                                            dim=1)  # bx3d
            else:
                batch_graph_emb = torch.cat([batch_graph_emb_by_node, batch_graph_emb_by_edge], dim=1)  # bx2d
            batch_graph_emb = self.dense_product_graph_level(batch_graph_emb)
            batch_graph_emb = torch.nn.functional.elu(batch_graph_emb)  # bxd
            return node_emb, batch_graph_emb  # bnxd, bxd


class RCIA2CNet(torch.nn.Module):
    def __init__(self, hidden_dimension, num_egat_heads, num_egat_layers, residual=True, have_fp=True):
        super(RCIA2CNet, self).__init__()

        self.base_encoder = BaseRCIEncoder(hidden_dimension=hidden_dimension,
                                           num_egat_heads=num_egat_heads,
                                           num_egat_layers=num_egat_layers,
                                           residual=residual, have_fp=have_fp)
        # policy
        self.dense_policy0 = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.dense_policy1 = torch.nn.Linear(hidden_dimension, 1)

        # value
        self.dense_value0 = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.dense_value1 = torch.nn.Linear(hidden_dimension, 1)

    def policy(self, batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask, split=False):
        # bnx1,   bxd
        node_emb, batch_graph_emb = self.base_encoder(batch_product_dgl, batch_placed_rc, batch_product_fp)

        list_node_embs = torch.split(node_emb, batch_product_dgl.batch_num_nodes().tolist())  # [nxd, nxd, ...]
        rst = []
        for idx, one_node_embs in enumerate(list_node_embs):
            rst.append(one_node_embs)
            rst.append(batch_graph_emb[idx].reshape(1, -1))
        rst = torch.cat(rst, dim=0)  # b(n+1)xd

        rst = self.dense_policy0(rst)  # b(n+1)xd
        rst = torch.nn.functional.elu(rst)  # b(n+1)xd
        rst = self.dense_policy1(rst)  # b(n+1)x1

        # mask
        rst = rst + batch_mask * -1e9  # b(n+1)x1

        rst = torch.split(rst, (batch_product_dgl.batch_num_nodes() + 1).tolist())  # [(n+1)x1, (n+1)x1, ...]
        rst = [torch.nn.functional.softmax(_rst, dim=0) for _rst in rst]  # [(n+1)x1, (n+1)x1, ...]
        if split:
            return rst  # [(n+1)x1, (n+1)x1, ...]
        else:
            rst = torch.cat(rst, dim=0)  # b(nx1)x1
            return rst

    def value(self, batch_product_dgl, batch_placed_rc, batch_product_fp):
        # bnx1,   bxd
        node_emb, batch_graph_emb = self.base_encoder(batch_product_dgl, batch_placed_rc, batch_product_fp)

        rst = self.dense_value0(batch_graph_emb)  # bxd
        rst = torch.nn.functional.elu(rst)  # bxd
        rst = self.dense_value1(rst)  # bx1
        return rst



