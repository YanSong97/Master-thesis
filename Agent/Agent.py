
import torch
import numpy as np

import math

import torch.nn as nn




class Agent:
    def __init__(self, env_case, state_dim=4, action_dim=1, model='LLB', deterministic=True, device='cuda',
                 rand_seed=1):

        self.env = CartPoleModEnv(case=env_case)
        self.env_case = env_case
        # self.env = gym.make('CartPole-BT-dH-v0')
        # self.env = gym.make('CartPoleMod-v2')

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.observations_list = []
        self.actions_list = []
        self.MSEloss = nn.MSELoss()
        self.model_name = model

        if model == 'DRNN':
            self.model = DRNN(action_dim, 32, state_dim, device, 'LSTM').to(device)
        elif model == 'SRNN':
            self.model = SRNN(action_dim, 32, state_dim, device, 'LSTM',
                              noise=0.5 * torch.tensor([0.1, 0.1, 0.1, 1])).to(device)
        elif model == 'LLB':
            self.model = BRNN(action_dim, 32, state_dim, device, 'LSTM').to(device)

        self.model_optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.mseloss = nn.MSELoss()
        self.model_training_loss_list = []

        self.policy = controller(action_dim, state_dim, device).to(device)
        self.deterministic_policy = deterministic

        # np.random.seed(rand_seed)
        # self.env.seed(rand_seed)
        # torch.manual_seed(rand_seed)

    def env_rollout(self, if_remember, plan, behaviour_uncertainty):
        """
        interaction with the environment using the current policy.
        """
        done = False
        state = self.env.reset()
        total_reward = 0
        i = 0
        temp_obs_list = []
        temp_actions_list = []

        if if_remember:
            temp_obs_list.append(torch.tensor(state))

        while not done:
            i += 1

            if plan == 'random':
                action = self.env.action_space.sample()
            elif plan == 'pg' or 'rp':

                state_tensor = torch.tensor(np.vstack(state)).float().squeeze()

                action = self.policy.make_decision(state_tensor.to(self.device), behaviour_uncertainty)
                action = action.detach().cpu().numpy()


            else:
                raise NotImplementedError
            # print('action = ', action)

            # next_state, reward, done, _  = self.env.step(int(action))
            next_state, reward, _, _ = self.env.step(action)

            # print('next state = ', next_state)

            done = next_state[0] < -2.4 \
                   or next_state[0] > 2.4 \
                   or next_state[2] < -12 * 2 * math.pi / 360 \
                   or next_state[2] > 12 * 2 * math.pi / 360 \
                   or i >= 200

            # print('done = ', done)

            if if_remember:
                temp_obs_list.append(torch.tensor(np.vstack(next_state)).squeeze())

                # temp_obs_list.append(torch.tensor(next_state))

                temp_actions_list.append(torch.tensor(action).float())

            state = next_state
            total_reward += 1

            # if total_reward > 200:
            #    break

        if if_remember:
            self.observations_list.append(torch.stack(temp_obs_list).float())  # list of shape [seq, output]
            self.actions_list.append(torch.stack(temp_actions_list).float())  # list of shape [seq-1, 1]

        return total_reward

    def model_learning(self, num_epoch, num_batch):
        """
        perform model leanring using data self.observation_list and self.actions_list; since the data has variable length, one could
        try truncate the data into same length or pack_padded_sequence, but here we would simply train each single sample in a batch,
        and during each epoch, the parameter is only updated once using part of the dataset
        num_epoch : number of training epoch
        num_batch: this is actually number of samples we want the model to be trained on during each epoch
        """

        for e in range(num_epoch):
            self.model_optimiser.zero_grad()

            idx = np.random.choice(len(self.observations_list), num_batch)
            trun_obs = truncate_sequence([self.observations_list[i] for i in idx], batch_first=False)
            trun_actions = truncate_sequence([self.actions_list[j] for j in idx], batch_first=False)

            if self.model_name == 'DRNN':
                pred = self.model(trun_obs[0, :, :], trun_actions)
            elif self.model_name == 'SRNN':
                pred, _, _ = self.model(trun_obs[0, :, :], trun_actions)
            elif self.model_name == 'LLB':
                N = int(trun_actions.numel())
                b_FE, b_LL, b_KL = self.model(trun_obs, trun_actions, N)
                loss = -b_FE
            if self.model_name is not 'LLB':
                loss = self.MSEloss(torch.cat(pred), trun_obs[1:, :, :].to('cuda'))
            loss.backward()

            # for i in idx:
            #    training_obs = self.observations_list[i].unsqueeze(1)       #[seq, 1, output]
            #    training_actions = self.actions_list[i]                      #[seq-1, 1]

            #    pred = self.model(training_obs[0,:,:], training_actions.unsqueeze(-1))
            #    loss = self.mseloss(torch.cat(pred).unsqueeze(1), training_obs[1:, :, :].to(self.device))
            #    temp_loss += loss
            # temp_loss.backward()

            self.model_optimiser.step()
            self.model_training_loss_list.append(loss.item())

            if e % 1000 == 0:
                if self.model_name == 'LLB':
                    print('Epoch{}; FE = {}; LL = {}; KL = {}.'.format(e, b_FE.item(), b_LL.item(), b_KL.item()))
                else:
                    print('Epoch:{}; loss = {}.'.format(e, loss.item()))

    def cost(self, state):
        """
        cost = 5*angle^2 + position^2
        state : [seq, batch, output]
        return [seq, batch, 1]
        """
        return (5 * state[:, :, 2] ** 2 + state[:, :, 0] ** 2).unsqueeze(-1)  # [seq, batch, 1]

    '''
    def cost(self, states, sigma=0.25):
        """
        states : [seq, batch, output]
        return : [seq, batch, 1]
        """
        l = 0.6
        seq_length = states.size(0)
        batch_size = states.size(1)
        feature_dim = states.size(-1)

        goal = Variable(torch.FloatTensor([0.0, l])).unsqueeze(0).unsqueeze(0).expand(seq_length,1, 2).to(self.device)     #[seq, 1,2]

        # Cart position
        cart_x = states[:,:, 0]         #[seq, batch]
        # Pole angle
        thetas = states[:,:,2]          #[seq, bnatch]
        # Pole position
        x = torch.sin(thetas)*l         #[seq, batch]
        y = torch.cos(thetas)*l
        positions = torch.stack([cart_x + x, y], -1)             #[seq, batch, 2]


        squared_distance = torch.sum((goal - positions)**2, -1).unsqueeze(-1)          #[]

        squared_sigma = sigma**2
        cost = 1 - torch.exp(-0.5*squared_distance/squared_sigma)

        return cost
    '''

    def policy_learning(self, imagine_num, num_particle, num_epoch, batch_size, horizon, plan, w_uncertainty,
                        e_uncertainty, plot=False):
        """
        we utilise the current learned model to do policy learning on imagined data
        num_epoch : number of epochs we want to run our policy gradient for
        batch_size : number of samples we want to train the policy on/ number of initial states

        we creat batch_size number of initial state, the model then rollout for a fixed length(horizon), the sum of cost for each
        imagined trajectory is computed, from which the gradient is taken w.r.t the policy parameters
        """
        # creat inital states
        for i in range(imagine_num):
            # initial_state = []
            # for b in range(batch_size):
            #    init_x = self.env.reset()
            #    initial_state.append(torch.tensor(init_x).float())
            # initial_state = torch.stack(initial_state)          #[batch, output]
            initial_state = torch.tensor(self.env.reset()).unsqueeze(0).float()  # [1, output]
            if plot:
                initial_state = torch.zeros_like(initial_state)
            # initial_state = torch.tensor(np.array([ 0.04263216,  0.00452452, -0.03763419, -0.03992425])).float().unsqueeze(0)

            # learn the policy parameter using current model

            model_f = self.model.imagine

            if plan == 'pg':
                policy_train_loss = self.policy.pg_train(num_epoch, num_particle, initial_state, horizon, self.cost,
                                                         model_f,
                                                         w_uncertainty, e_uncertainty, gamma=1)
            elif plan == 'rp':
                policy_train_loss = self.policy.rp_train(num_epoch, num_particle, initial_state, horizon, self.cost,
                                                         model_f,
                                                         w_uncertainty, e_uncertainty, gamma=1)
        self.policy_loss = policy_train_loss
        """
        total_reward = []
        for i in range(20):
            init_x = torch.tensor(self.env.reset()).unsqueeze(0).float() 

            imagine_reward = self.model.validate_by_imagination(init_x, self.policy.forward, 
                                                            plan, w_uncertainty, e_uncertainty)
            total_reward.append(imagine_reward)
            #print('temp reward', imagine_reward)
        mean_reward = np.mean(total_reward)
        std_reward = np.std(total_reward)
        print('Training reward: mean {}, std {}.'.format(mean_reward, std_reward))
        return mean_reward, std_reward

        """
        """
        total_cost10 = []
        total_cost100= []
        for i in range(20):
            initial_state = torch.tensor(self.env.reset()).unsqueeze(0).float()         #[1, output]
            mean_cost10 = self.policy.rp_validate(num_particle, initial_state, 10, self.cost, model_f, w_uncertainty, e_uncertainty)
            mean_cost100 = self.policy.rp_validate(num_particle, initial_state, 100, self.cost, model_f, w_uncertainty, e_uncertainty)

            total_cost10.append(mean_cost10)
            total_cost100.append(mean_cost100)

        mean_cost10 = np.mean(total_cost10)
        std_cost10 = np.std(total_cost10)

        mean_cost100 = np.mean(total_cost100)
        std_cost100 = np.std(total_cost100)


        return mean_cost10, std_cost10, mean_cost100, std_cost100

        """
        """
        total_cost10 = []
        for i in range(20):
            initial_state = torch.tensor(self.env.reset()).unsqueeze(0).float()         #[1, output]
            mean_cost = self.policy.rp_validate(num_particle, initial_state, 10, self.cost, model_f, w_uncertainty, e_uncertainty, gamma=1)
            total_cost10.append(mean_cost)

        mean_cost10 = np.mean(total_cost10)
        std_cost10 = np.std(total_cost10)

        total_cost100 = []
        for i in range(20):
            initial_state = torch.tensor(self.env.reset()).unsqueeze(0).float()         #[1, output]
            mean_cost = self.policy.rp_validate(num_particle, initial_state, 100, self.cost, model_f, w_uncertainty, e_uncertainty, gamma=1)
            total_cost100.append(mean_cost)

        mean_cost100 = np.mean(total_cost100)
        std_cost100 = np.std(total_cost100)


        return mean_cost10, std_cost10, mean_cost100, std_cost100
        """
