import copy
import json
import torch
import random
from RetroUtils.smiles2dgl import smiles_to_dglGraph


class RCIEnv:
    def __init__(self, max_len=4):  # RawDataFile_path='../data/RawData'

        self.max_len = max_len

    def reset(self, data_dict):
        self.data_dict = data_dict
        self.rxn = self.data_dict['rxn']
        self.product = self.data_dict['product']
        self.reactants = self.data_dict['reactants']
        self.label_RcNodeIdx = self.data_dict['rc_node_idx']

        # state
        self.product_dgl = smiles_to_dglGraph(smiles=self.product, seed=0, add_self_loop=False, Conformer=False)
        self.num_nodes_product = self.product_dgl.num_nodes()
        self.product_fp = torch.tensor(self.data_dict['product_fp'], dtype=torch.float32).reshape(1, -1)
        Placed_RcNodeIdx = torch.zeros(self.num_nodes_product)  # n
        RcNodeIdx_list = []

        # mask
        mask = torch.zeros(self.num_nodes_product + 1)  # (n+1)
        mask[-1] = 1
        mask = mask.reshape(-1, 1)  # (n+1)x1

        t = 0

        self.state = [self.product_dgl, self.product_fp,
                      Placed_RcNodeIdx, RcNodeIdx_list,
                      mask, t]
        return copy.deepcopy(self.state)

    def compute_Placed_tensor(self, num_nodes, node_set):  # num_nodes: int, node_set: [idx, idx, ...]
        assert -1 not in node_set, '-1 in node_set'
        placed = torch.zeros(num_nodes)  # n
        placed[node_set] = 1
        return placed  # n

    def compute_graph_one_hop_neighbor(self, node_set):  # node_set: [idx, idx, ...]
        one_hop_neighbor = set()
        for node in node_set:
            one_hop_nodes = self.product_dgl.successors(node)
            one_hop_neighbor = one_hop_neighbor | set(one_hop_nodes.tolist())
        one_hop_neighbor = one_hop_neighbor - set(node_set)
        one_hop_neighbor = list(one_hop_neighbor)
        one_hop_neighbor.sort()
        return one_hop_neighbor  # [idx, idx, ...]

    def compute_mask(self, node_set):  # ode_set: [idx, idx, ...], tag: int
        if len(node_set) == 0:
            mask = torch.zeros(self.num_nodes_product + 1)  # (n+1)
            mask[-1] = 1
            mask = mask.reshape(-1, 1)  # (n+1)x1
        else:
            assert -1 not in node_set, '-1 in node_set'
            one_hop_neighbor = self.compute_graph_one_hop_neighbor(node_set)  # [idx, idx, ...]
            if len(one_hop_neighbor) != 0:
                assert max(
                    one_hop_neighbor) < self.num_nodes_product, f'error at one_hop_neighbor, one_hop_neighbor: {one_hop_neighbor}, num_nodes_product: {self.num_nodes_product}'
            mask = torch.ones(self.num_nodes_product + 1)  # (n+1)
            mask[one_hop_neighbor] = 0
            mask[-1] = 0
            mask = mask.reshape(-1, 1)  # (n+1)x1
        return mask

    def step(self, action):
        if action != -1:  # not rc-stop-action
            assert action < self.num_nodes_product, 'action out of num_nodes_product'
            # change: Placed_RcNodeIdx, RcNodeIdx_list, mask, t
            RcNodeIdx_list = copy.deepcopy(self.state[3])  # [idx, idx, ...]
            assert action not in RcNodeIdx_list, f'action: {action} in RcNodeIdx_list: {RcNodeIdx_list}'
            RcNodeIdx_list.append(action)
            RcNodeIdx_list.sort()  # !
            Placed_RcNodeIdx = self.compute_Placed_tensor(num_nodes=self.num_nodes_product,
                                                          node_set=RcNodeIdx_list)  # !

            mask = self.compute_mask(node_set=RcNodeIdx_list)  # !
            t = copy.deepcopy(self.state[-1]) + 1  # !

            # next_state
            next_state = [self.product_dgl, self.product_fp,
                          Placed_RcNodeIdx, RcNodeIdx_list,
                          mask, t]

            if next_state[-1] > self.max_len:
                done = 1
                if set(RcNodeIdx_list) == set(self.label_RcNodeIdx):
                    reward = 1
                else:
                    reward = 0
            else:
                done = 0
                reward = 0

            self.state = copy.deepcopy(next_state)
            return reward, copy.deepcopy(next_state), done

        elif action == -1:  # rc-stop-action
            # change: mask, t
            mask = torch.zeros(self.num_nodes_product + 1)
            mask = mask.reshape(-1, 1)  # (n+1)x1  # !
            t = copy.deepcopy(self.state[-1]) + 1  # !

            # not change: product_dgl, product_fp, Placed_RcNodeIdx, RcNodeIdx_list
            Placed_RcNodeIdx = copy.deepcopy(self.state[2])
            RcNodeIdx_list = copy.deepcopy(self.state[3])

            next_state = [self.product_dgl, self.product_fp,
                          Placed_RcNodeIdx, RcNodeIdx_list,
                          mask, t]

            done = 1
            if set(RcNodeIdx_list) == set(self.label_RcNodeIdx):
                reward = 1
            else:
                reward = 0

            self.state = copy.deepcopy(next_state)
            return reward, copy.deepcopy(next_state), done

    def generate_random_gt_trajectory(self):
        label_RcNodeIdx = copy.deepcopy(self.label_RcNodeIdx)  # [idx, idx, ...]
        rc_trajectory = []

        # rc
        action = random.choice(label_RcNodeIdx)
        rc_trajectory.append(action)
        label_RcNodeIdx.pop(label_RcNodeIdx.index(action))
        if len(label_RcNodeIdx) != 0:
            while True:
                one_hop_neighbor = self.compute_graph_one_hop_neighbor(rc_trajectory)
                candidate_nodes = set(one_hop_neighbor) & set(label_RcNodeIdx)
                action = random.choice(list(candidate_nodes))
                rc_trajectory.append(action)
                label_RcNodeIdx.pop(label_RcNodeIdx.index(action))
                if len(label_RcNodeIdx) == 0:
                    rc_trajectory.append(-1)
                    break
        elif len(label_RcNodeIdx) == 0:
            rc_trajectory.append(-1)

        return rc_trajectory

    def to_dict(self):
        return vars(self)


