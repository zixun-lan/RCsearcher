import os
import json
import tqdm
import torch
import numpy as np
from RCIA2CNet import RCIA2CNet, collate_RCI
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RCIAgentPPO(torch.nn.Module):
    def __init__(self, hidden_dimension, num_egat_heads, num_egat_layers,
                 residual=True, have_fp=True, gamma=0.99, eps_clip=0.2,
                 value_coefficient=0.5, entropy_coefficient=0.01,
                 learning_rate=0.001, device=torch.device('cpu')):
        super(RCIAgentPPO, self).__init__()

        self.RCIA2CNet = RCIA2CNet(hidden_dimension=hidden_dimension,
                                   num_egat_heads=num_egat_heads,
                                   num_egat_layers=num_egat_layers,
                                   residual=residual, have_fp=have_fp)
        self.RCIA2CNet.to(device)

        self.optimizer = torch.optim.Adam(self.RCIA2CNet.parameters(), lr=learning_rate)
        self.mseLoss = torch.nn.MSELoss(reduction='none')
        self.bceLoss = torch.nn.BCELoss()
        self.device = device

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

        self.buffer = []

    def store_transition(self, transition):
        # transition = (state, action, p_of_a, r, next_state, done)  # ([], int, float, float, [], int)
        self.buffer.append(transition)

    def clear_buffer(self):
        self.buffer = []

    def save_param(self, save_path):
        torch.save(self.RCIA2CNet.state_dict(), save_path)
        print('save success!')

    def load_param(self, ckpt_path):
        self.RCIA2CNet.load_state_dict(torch.load(ckpt_path))
        print('load success!')

    def select_action(self, one_state):
        # state = [product_dgl, product_fp, Placed_RcNodeIdx, RcNodeIdx_list, mask, t]
        batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = collate_RCI([one_state])
        batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = batch_product_dgl.to(self.device), \
                                                                           batch_placed_rc.to(self.device), \
                                                                           batch_product_fp.to(self.device), \
                                                                           batch_mask.to(self.device)

        self.RCIA2CNet.eval()
        with torch.no_grad():
            # (n+1)x1
            rst_policy = self.RCIA2CNet.policy(batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask,
                                               split=False).reshape(-1)

        dist = Categorical(rst_policy)
        action = dist.sample().item()

        if len(rst_policy) - 1 == action:
            action = -1

        p_of_action = rst_policy[action].item()
        return action, p_of_action, rst_policy

    def update(self, num_epochs, batch_size, imitation=False):
        if imitation:
            # (state, action, p_of_a, r, next_state, done)  # ([], int, float, float, [], int)
            list_state, list_action, list_p_of_a, list_r, list_next_state, list_done = map(list, zip(*self.buffer))
            buffer_action = torch.tensor(list_action, dtype=torch.long).reshape(-1, 1)  # Mx1

            self.RCIA2CNet.train()
            loss_list = []
            for _ in tqdm.tqdm(range(num_epochs), leave=False):
                for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batch_size=batch_size,
                                          drop_last=False):
                    # for i in range(len(self.buffer)):
                    #     index = [i]

                    # batch_state
                    samples_states = [list_state[i] for i in index]  # [[], [], ...]
                    batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = collate_RCI(samples_states)
                    batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = \
                        batch_product_dgl.to(self.device), \
                        batch_placed_rc.to(self.device), \
                        batch_product_fp.to(self.device), \
                        batch_mask.to(self.device)

                    # new_p
                    batch_action = buffer_action[index]  # bx1
                    list_policy = self.RCIA2CNet.policy(batch_product_dgl, batch_placed_rc, batch_product_fp,
                                                        batch_mask,
                                                        split=True)  # [(n+1)x1, (n+1)x1, ...]
                    batch_new_p = []
                    for idx, action in enumerate(batch_action):  # action shape= 1
                        one_policy = list_policy[idx]
                        new_p = one_policy[action]  # 1x1
                        batch_new_p.append(new_p)
                    batch_new_p = torch.cat(batch_new_p, dim=0).reshape(-1, 1)  # bx1
                    assert batch_new_p.requires_grad, 'batch_new_p do not have grad!'

                    # label
                    label = torch.ones_like(batch_new_p).detach()

                    loss = self.bceLoss(batch_new_p, label)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.RCIA2CNet.parameters(), 1)
                    self.optimizer.step()
                    loss_list.append(loss.cpu().detach().item())
            return np.mean(loss_list)

        else:
            # (state, action, p_of_a, r, next_state, done)  # ([], int, float, float, [], int)
            list_state, list_action, list_p_of_a, list_r, list_next_state, list_done = map(list, zip(*self.buffer))
            buffer_action = torch.tensor(list_action, dtype=torch.long).reshape(-1, 1)  # Mx1
            buffer_p_of_a = torch.tensor(list_p_of_a, dtype=torch.float32).reshape(-1, 1)  # Mx1
            buffer_r = torch.tensor(list_r, dtype=torch.float32).reshape(-1, 1)  # Mx1
            buffer_done = torch.tensor(list_done, dtype=torch.float32).reshape(-1, 1)  # Mx1

            Gt = []
            discounted_r = 0
            for reward, d in zip(reversed(buffer_r), reversed(buffer_done)):
                if d:
                    discounted_r = 0
                discounted_r = reward + self.gamma * discounted_r
                Gt.insert(0, discounted_r)  # insert in front, cannot use append

            buffer_Gt = torch.tensor(Gt, dtype=torch.float32).reshape(-1, 1)  # Mx1

            loss_list = []
            self.RCIA2CNet.train()
            for _ in tqdm.tqdm(range(num_epochs), leave=False):
                for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batch_size=batch_size,
                                          drop_last=False):
                    # for i in range(len(self.buffer)):
                    #     index = [i]

                    # batch_state
                    samples_states = [list_state[i] for i in index]  # [[], [], ...]
                    batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = collate_RCI(samples_states)
                    batch_product_dgl, batch_placed_rc, batch_product_fp, batch_mask = \
                        batch_product_dgl.to(self.device), \
                        batch_placed_rc.to(self.device), \
                        batch_product_fp.to(self.device), \
                        batch_mask.to(self.device)

                    # batch_Gt
                    batch_Gt = buffer_Gt[index]  # bx1
                    if len(batch_Gt) > 1:
                        batch_Gt = (batch_Gt - batch_Gt.mean()) / (batch_Gt.std() + 1e-5)
                    batch_Gt = batch_Gt.to(self.device)  # bx1

                    # pre_v
                    batch_pre_v = self.RCIA2CNet.value(batch_product_dgl, batch_placed_rc, batch_product_fp)  # bx1

                    # advantage
                    batch_advantage = batch_Gt - batch_pre_v
                    batch_advantage = batch_advantage.detach()  # bx1

                    # old_p
                    batch_old_p = buffer_p_of_a[index].detach()  # bx1
                    batch_old_p = batch_old_p.to(self.device)  # bx1

                    # new_p
                    batch_action = buffer_action[index]  # bx1
                    list_policy = self.RCIA2CNet.policy(batch_product_dgl, batch_placed_rc, batch_product_fp,
                                                        batch_mask,
                                                        split=True)  # [(n+1)x1, (n+1)x1, ...]
                    batch_new_p = []
                    for idx, action in enumerate(batch_action):  # action shape= 1
                        one_policy = list_policy[idx]
                        new_p = one_policy[action]  # 1x1
                        batch_new_p.append(new_p)
                    batch_new_p = torch.cat(batch_new_p, dim=0).reshape(-1, 1)  # bx1
                    assert batch_new_p.requires_grad, 'batch_new_p do not have grad!'

                    # entropy
                    entropy = []
                    for _policy in list_policy:  # _policy (n+1)x1
                        _dist = Categorical(_policy.reshape(1, -1))
                        _entropy = _dist.entropy()  # shape = 1
                        entropy.append(_entropy)
                    entropy = torch.cat(entropy).reshape(-1, 1)  # bx1
                    assert entropy.requires_grad, 'entropy do not have grad!'

                    # loss
                    ratio = torch.exp(torch.log(batch_new_p) - torch.log(batch_old_p))  # a/b == exp(log(a)-log(b))  bx1

                    surr1 = ratio * batch_advantage  # bx1
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantage  # bx1
                    # loss = -torch.min(surr1, surr2).mean() + \
                    #        self.value_coefficient * (self.mseLoss(batch_pre_v, batch_Gt.detach()).mean()) - \
                    #        self.entropy_coefficient * (entropy.mean())

                    loss = -torch.min(surr1, surr2) + \
                           self.value_coefficient * self.mseLoss(batch_pre_v, batch_Gt.detach()) - \
                           self.entropy_coefficient * entropy
                    loss = loss.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.RCIA2CNet.parameters(), 1)
                    self.optimizer.step()
                    loss_list.append(loss.cpu().detach().item())
            return np.mean(loss_list)


