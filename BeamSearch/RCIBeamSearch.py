import copy
import json
import math
import time

from config import Config

import torch
import tqdm
from rdkit.Chem import MolFromSmiles, MolFromSmarts

from RCIAgent import RCIAgentPPO
from RCIEnv import RCIEnv


class RCIBeamSearch():
    def __init__(self, rciagent, k):
        self.agent = rciagent
        self.k = k

    def length_penalty(self, length, alpha=0.6):
        """
        计算长度惩罚值。

        Args:
        length (int): 生成的序列的长度。
        alpha (float): 惩罚的超参数。通常在 [0, 1] 之间，alpha 越大，惩罚越强。

        Returns:
        float: 长度惩罚值。
        """
        return math.pow((5 + length) / 6, alpha)

    def search(self, retro_env, data_dict, unique=False):
        state = retro_env.reset(data_dict=data_dict)
        # print(retro_env.label_RcNodeIdx)
        init_log_p = 0
        init_done = 0
        candidate = [(init_log_p, retro_env, init_done)]

        while sum([i[2] for i in candidate]) != len(candidate):
            # print('\nCANDIDATE', [(i[0], i[1].state[3], i[2]) for i in candidate])
            tmp_candi = copy.deepcopy(candidate)
            candidate = []
            for candi in tmp_candi:
                if candi[2]:
                    candidate.append(candi)
                    continue

                last_log_p = candi[0]
                action, p_of_action, rst_policy = self.agent.select_action(candi[1].state)
                all_p, all_actions = torch.sort(rst_policy, descending=True)
                all_actions = all_actions.tolist()
                all_p = all_p.tolist()
                k_actions = all_actions[:self.k]
                k_p = all_p[:self.k]

                # print('For state', candi[1].state[3], 'we have K actions', k_actions, 'K possibilities', k_p)
                k_env = [copy.deepcopy(candi[1]) for _ in range(k)]
                for i, act in enumerate(k_actions):
                    if k_p[i] == 0:
                        continue
                    if act == len(all_actions) - 1:
                        act = -1
                    r, next_state, done = k_env[i].step(act)
                    # print('with action', act, 'next state', r, next_state[3], done, 'current p', k_p[i])

                    candidate.append(
                        (last_log_p + math.log(k_p[i]) / self.length_penalty(len(next_state[3])), k_env[i], done))
            # print('new CANDIDATE', [(i[0], i[1].state[3], i[2]) for i in candidate])
            candidate = sorted(candidate, key=lambda x: x[0], reverse=True)[:self.k]
            # print('we choose k candidates', [(i[0], i[1].state[3]) for i in candidate])

        p_list = []
        idx_list = []
        for (p, env, done) in candidate:
            if env.state[3] in idx_list and unique:
                continue
            else:
                idx_list.append(env.state[3])
                p_list.append(p)
        rst = [i for i in zip(p_list, idx_list)]
        return rst



