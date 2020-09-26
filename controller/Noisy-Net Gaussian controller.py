import torch
import numpy as np

import torch.nn as nn



class controller(nn.Module):
    def __init__(self, action_dim=1, state_dim=4, deterministic=True, device='cuda'):
        super(controller, self).__init__()
        self.action_dim = action_dim
        controller_hid = 16
        self.state_dim = state_dim
        init_w = 1e-3

        self.linear = nn.Sequential(
            nn.Linear(state_dim, controller_hid),
            nn.ReLU(),
            nn.Linear(controller_hid, controller_hid),
            nn.ReLU()
        )

        self.W_mu = nn.Parameter(torch.zeros(action_dim, controller_hid + 1))
        self.W_logvar = nn.Parameter(torch.rand(action_dim, controller_hid + 1))

        self.W_logvar.data.fill_(np.log(0.5))

        # if not deterministic:
        #    self.std = 0.1

        # self.deterministic = deterministic

        # nn.Linear(self.state_dim, self.action_dim)        #output a p_logits for action 1
        self.device = device
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

    def reset_W(self, batch_size):
        self.W_sample = self.stack_W(batch_size, self.W_mu, torch.exp(self.W_logvar))

    def forward(self, state):
        """
        Given states input [batch, state_dim],
        """
        state = state.to(self.device)
        batch_size = state.size(0)

        # W = self.stack_W(batch_size, self.W_mu, torch.exp(self.W_logvar))       #[batch, action, state+1]
        W = self.W_sample

        x = self.linear(state)

        state1 = torch.cat([x, torch.ones(batch_size, 1).to(self.device)], dim=-1)  # [batch, state + 1]

        # print('W dim', W.shape)
        # print('state1 dim', state1.shape)
        action = torch.bmm(W, state1.unsqueeze(-1))  # [batch, action, 1]

        return torch.tanh(action).squeeze(-1), 0  # [batch, action]

        # out_mean = self.linear(state)         #[batch, action_dim]

        # if not self.deterministic:
        #    eps = torch.rand_like(out_mean).normal_().to(self.device)           #[batch, action_dim]
        #    out = out_mean + self.std * eps
        # else:
        #    out = out_mean

        # if len(out.shape) == 1:
        #    out = torch.clamp(out, -1, 1)
        # else:
        #    out = torch.clamp(out[:,0], -1, 1).unsqueeze(1)             #[1, batch, 1]

        # return torch.tanh(out)

        # if len(out.shape) == 1:
        #    return torch.clamp(out, -1, 1)
        # else:
        #    clamp_out = torch.clamp(out[:, 0], -1, 1).unsqueeze(-1)
        #    return clamp_out

    def make_decision(self, state, behaviour_uncertainty):
        """
        given a state [batch, state_dim], output a action
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if behaviour_uncertainty:
            self.reset_W(1)
        else:
            self.W_sample = self.W_mu.unsqueeze(0)

        a, _ = self.forward(state)

        return a.detach().squeeze(-1)

        # out_mean = self.linear(state)
        # if not self.deterministic and behaviour_uncertainty:
        #    eps = torch.rand_like(out_mean).normal_().to(self.device)
        #    out = out_mean + self.std * eps
        # else:
        #    out = out_mean
        # if len(out.shape) == 1:

        #    out = torch.clamp(out, -1, 1)
        # else:
        #    out = torch.clamp(out[:,0], -1, 1)
        # return out.detach()
        # return torch.tanh(out)

    def pg_train(self, num_epoch, initial_state, horizon, cost_f, model_imagine_f, gamma=0.95):
        """
        initial_state : [batch, state_dim]

        """
        loss_list = []
        num_particle = 100
        initial_state = initial_state.expand(num_particle, self.state_dim)

        for e in range(num_epoch):
            self.optimiser.zero_grad()

            output_matrix, action_log_prob_matrix = model_imagine_f(initial_state, self.forward, horizon, plan='pg')
            cost = cost_f(output_matrix).detach()  # [seq-1, batch, 1]

            cost = cost * torch.tensor([gamma ** (t + 1) for t in range(cost.size(0))]).unsqueeze(-1).unsqueeze(-1).to(
                self.device)

            # baseline = torch.mean(cost, dim = 0).unsqueeze(0)       #[1, batch, 1]
            # cost = cost - torch.mean(cost, dim = 0).unsqueeze(0)
            # loss = ((cost-baseline) * action_log_prob_matrix).sum(0)
            loss = cost.sum(0) * action_log_prob_matrix.sum(0)

            loss = loss.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5)

            self.optimiser.step()
            loss_list.append(loss.item())
            if e % 50 == 0:
                print('Epoch = {}; Policy gradient training loss = {}'.format(e, loss.item()))
        return loss_list

    def rp_train(self, num_epoch, initial_state, horizon, cost_f, model_imagine_f, gamma=0.9):
        """
        From an initial state, use mode imagination function to make prediction of the next state accordin to the action proposed by
        the controller, we fixed the horizon and compute the total reward of the trajectory, from which the gradient w.r.t policy
        parameters is taken.
        inital_state: [batch, output_dim]
        """
        loss_list = []
        num_particle = 100
        initial_state = initial_state.expand(num_particle, self.state_dim)

        for e in range(num_epoch):
            self.reset_W(num_particle)
            self.optimiser.zero_grad()

            output_matrix, _ = model_imagine_f(initial_state, self.forward, horizon,
                                               plan='rp')  # [seq-1, batch, output], [seq-1, batch, 1]

            self.temp_output_matrix = torch.cat([initial_state.unsqueeze(0).to(self.device), output_matrix], dim=0)

            cost = cost_f(output_matrix)  # [seq-1, batch, 1]
            # multiply by discount factor
            # cost = cost *  ((torch.arange(cost.size(0)+1,1,-1).float()).unsqueeze(-1).unsqueeze(-1)/cost.size(0)
            #                    ).expand(cost.shape).float().to(self.device)

            cost = cost * torch.tensor([gamma ** (t + 1) for t in range(cost.size(0))]).unsqueeze(-1).unsqueeze(-1).to(
                self.device)

            loss = cost.sum()  # [batch, 1]
            # loss = torch.exp(action_log_prob_matrix.sum(0)) * cost.sum(0)
            # loss = (cost * action_log_prob_matrix).sum(0)                 #[batch, 1]

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5)

            self.optimiser.step()
            loss_list.append(loss.item())
            if e % 200 == 0:
                print('Epoch = {}; Policy gradient training loss = {}'.format(e, loss.item()))
        return loss_list

    def stack_W(self, batch_size, mean, sigma):
        list_of_W = []
        for i in range(batch_size):
            temp_W = mean + sigma * torch.rand_like(mean).normal_().to(self.device)
            # temp_W = self.reparametrise(mean, sigma).unsqueeze(0)

            list_of_W.append(temp_W.unsqueeze(0))
        return torch.cat(list_of_W)
