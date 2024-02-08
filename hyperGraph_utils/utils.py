import dhg
import torch
import itertools
import pickle


class OurHypergraph(dhg.Hypergraph):
    def __init__(self, num_v, e_list, x_tensor=None, batch_tag=0, batch_num_nodes=None):
        super().__init__(num_v=num_v, e_list=e_list)
        r"""
        Args:
            num_v (int): number of nodes
            e_list (list of tuple): hyperedges  [(),(),(),...]
            x_tensor (tensor): torch.tensor() node_feature
            batch_tag (int): 0 is non-batched hypergraph, q is batched hypergraph
            batch_num_nodes (tensor): torch.tensor([hg1.num_v, hg2.num_v, ...]) or None
        """

        self.batch_tag = batch_tag
        self.x_tensor = x_tensor
        self.batch_num_nodes = batch_num_nodes

    def our_to(self, device):
        if self.x_tensor is not None:
            self.x_tensor = self.x_tensor.to(device)
        if self.batch_num_nodes is not None:
            self.batch_num_nodes = self.batch_num_nodes.to(device)
        return self.to(device)


def batch_hypergraph(list_ourhypergraph):
    r"""
    batch_hypergraph batch list of OurHypergraph instances.
    :param list_ourhypergraph (list of OurHypergraph instances): [hg0, hg1, hg2, hg3]
    :return: OurHypergraph instance whose batch_tag is 1
    """

    # batch_nodes
    batch_num_nodes = [hg.num_v for hg in list_ourhypergraph]
    batch_total_nodes = sum(batch_num_nodes)

    # batch_edges
    tmp_batch_e_list = [hg.e[0] for hg in
                        list_ourhypergraph]  # [[(0, 1, 2), (2, 3), (0, 4)], [(0, 1, 2), (2, 3), (0, 4)],]
    batch_add_num = [0] + batch_num_nodes
    batch_add_num = batch_add_num[0:-1]
    batch_add_num = list(itertools.accumulate(batch_add_num))
    batch_e_list = []
    for idx, accumulate_add in enumerate(batch_add_num):
        one_e_list = tmp_batch_e_list[idx]
        batch_e_list = batch_e_list + [tuple(map(lambda x: x + accumulate_add, hyperedge)) for hyperedge in one_e_list]

    # batch_x_tensor
    batch_x_tensor = [hg.x_tensor for hg in list_ourhypergraph]
    batch_x_tensor = torch.cat(batch_x_tensor, dim=0)

    BatchHypergraph = OurHypergraph(num_v=batch_total_nodes,
                                    e_list=batch_e_list,
                                    x_tensor=batch_x_tensor,
                                    batch_tag=1,
                                    batch_num_nodes=torch.tensor(batch_num_nodes))
    return BatchHypergraph


def save_hypergraph(save_path, save_data):
    with open(save_path, "wb") as fp:
        pickle.dump(save_data, fp)
    print('save_hypergraph succeed! ')


def load_hypergraph(load_path):
    with open(load_path, "rb") as fp:
        data = pickle.load(fp)
    print('load_hypergraph succeed! ')
    return data



