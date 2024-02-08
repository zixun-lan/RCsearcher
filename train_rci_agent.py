import os
import json
import tqdm
import torch
import random
import pprint
import numpy as np
from config import Config
from RCIEnv import RCIEnv
from RCIAgent import RCIAgentPPO
from rdkit.Chem import MolFromSmiles, MolFromSmarts


def eval_success_rate(rci_env, agent, RawVal):
    test = random.sample(RawVal, 1000)
    success_list = []
    for idx_data, data_dict in tqdm.tqdm(enumerate(test), leave=False):
        state = rci_env.reset(data_dict=data_dict)
        done = 0
        total_reward = 0
        while not done:
            action, p_of_action, rst_policy = agent.select_action(state)
            r, next_state, done = rci_env.step(action)
            state = next_state
            total_reward += r
            if done:
                if set(state[3]) == set(rci_env.label_RcNodeIdx):
                    success_list.append(1)
                else:
                    success_list.append(0)
    success_rate = np.mean(success_list)
    return success_rate


def main(args, have_imitation=True, ckpt=None):
    print('configuration: ')
    pprint.pprint(args.to_dict())
    print('----------------------------------')

    dataset_name = args.dataset_name
    raw_path = os.path.join(args.RawDataFile_path, dataset_name)
    processed_path = os.path.join(args.ProcessedDataFile_path, dataset_name)

    print('start: process data')
    config_name = args.config_name
    save_root_path = os.path.join('./outputs_rci', config_name)
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    # save config
    args.save(os.path.join(save_root_path, config_name))

    with open(os.path.join(processed_path, 'raw_train.json'), 'r') as f:
        RawTrain = json.load(f)
    with open(os.path.join(processed_path, 'raw_val.json'), 'r') as f:
        RawVal = json.load(f)

    device = torch.device(args.device)
    rci_env = RCIEnv(max_len=args.trajectory_max_length)
    agent = RCIAgentPPO(hidden_dimension=args.hidden_dimension, num_egat_heads=args.num_egat_heads,
                        num_egat_layers=args.num_egat_layers, residual=args.residual, have_fp=args.have_fp,
                        gamma=args.gamma, eps_clip=args.eps_clip, value_coefficient=args.value_coefficient,
                        entropy_coefficient=args.entropy_coefficient, learning_rate=args.learning_rate, device=device)
    min_num_transitions = args.min_num_transitions
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # imitation learning
    if have_imitation:
        print('start: imitation learning')
        # imitation learning
        learning = 0
        for n_epi in range(args.num_imitation):
            random.shuffle(RawTrain)
            loss_list = []
            print('{}-epoch imitation start!***************************'.format(n_epi))
            for idx_data, data_dict in tqdm.tqdm(enumerate(RawTrain), leave=False):

                state = rci_env.reset(data_dict=data_dict)
                done = 0
                total_reward = 0

                gt_idx = 0
                gt = rci_env.generate_random_gt_trajectory()

                while not done:
                    action, p_of_action, rst_policy = agent.select_action(state)

                    action = gt[gt_idx]
                    gt_idx += 1
                    p_of_action = rst_policy[action].item()

                    r, next_state, done = rci_env.step(action)
                    trans = (state, action, p_of_action, r, next_state, done)
                    # (state, action, p_of_a, r, next_state, done)  # ([], int, float, float, [], int)
                    agent.store_transition(trans)
                    state = next_state
                    total_reward += r

                if len(agent.buffer) >= min_num_transitions:
                    print('------------------------------------')
                    print('start {}-th imitation learning at {} data in epoch {}.'.format(learning, idx_data, n_epi))

                    mean_loss = agent.update(num_epochs=num_epochs, batch_size=batch_size, imitation=True)
                    loss_list.append(mean_loss)

                    save_imitation_path = os.path.join(save_root_path, 'rci_imitation.pkl')
                    agent.save_param(save_imitation_path)

                    learning += 1
                    print('{}-th learning end, loss is {:.2f}.'.format(learning, mean_loss))
                    agent.clear_buffer()

            print('{}-epoch end! final loss_list: {} ***************************'.format(n_epi, np.mean(loss_list)))
        print('end: imitation learning')

    # ppo learning
    print('start: ppo learning')
    max_success_rate = -1
    learning = 0
    if ckpt is not None:
        agent.load_param(ckpt)
    for n_epi in range(args.total_num_epoch):
        random.shuffle(RawTrain)
        success_list = []
        loss_list = []
        print('{}-epoch start!***************************'.format(n_epi))
        for idx_data, data_dict in tqdm.tqdm(enumerate(RawTrain), leave=False):

            state = rci_env.reset(data_dict=data_dict)
            done = 0
            total_reward = 0

            while not done:
                action, p_of_action, rst_policy = agent.select_action(state)

                r, next_state, done = rci_env.step(action)
                trans = (state, action, p_of_action, r, next_state, done)
                # (state, action, p_of_a, r, next_state, done)  # ([], int, float, float, [], int)
                agent.store_transition(trans)
                state = next_state
                total_reward += r
                if done:
                    if set(state[3]) == set(rci_env.label_RcNodeIdx):
                        success_list.append(1)
                        print('success!!!!!!!!!!!!!!!!!')
                    else:
                        success_list.append(0)

            if len(agent.buffer) >= min_num_transitions:
                print('------------------------------------')
                print('start {}-th learning at {} data in epoch {}.'.format(learning, idx_data, n_epi))

                mean_loss = agent.update(num_epochs=num_epochs, batch_size=batch_size, imitation=False)
                loss_list.append(mean_loss)

                save_ppo_path = os.path.join(save_root_path, 'rci_ppo.pkl')
                agent.save_param(save_ppo_path)

                print('{}-th learning end, loss is {:.2f}.'.format(learning, mean_loss))
                print('recent 32 final success reward: {:.2f}'.format(np.mean(success_list[-32:])))
                print('test max_success_rate: ', max_success_rate)
                agent.clear_buffer()

                if learning % args.save_at_num_update == 0:
                    print('start eval!')
                    success_rate = eval_success_rate(rci_env=rci_env, agent=agent, RawVal=RawVal)
                    print('end eval!')
                    if success_rate > max_success_rate:
                        max_success_rate = success_rate
                        save_best_path = os.path.join(save_root_path, 'rci_best.pkl')
                        agent.save_param(save_best_path)
                        print('update best!!!!!!!!!!!!!!!', 'max_success_rate: ', max_success_rate)
                learning += 1

        print('{}-epoch end! success rate: {} loss: {} *******************'.format(n_epi, np.mean(success_list),
                                                                                   np.mean(loss_list)))
        print('test max_success_rate: ', max_success_rate, '*******************')
    print('end: ppo learning')


if __name__ == '__main__':
    args = Config(config_path='./ConfigFile/RCI-USPTO-50k.json')
    main(args, have_imitation=True)

