
import torch

import torch.nn as nn

from torch.distributions import Normal





class controller(nn.Module):
    def __init__(self, action_dim=1, state_dim=4, deterministic=True, device='cuda'):
        super(controller, self).__init__()
        self.action_dim = action_dim
        controller_hid = 16
        self.state_dim = state_dim
        init_w = 1e-3

        self.linear1 = nn.Linear(self.state_dim, controller_hid)
        self.linear2 = nn.Linear(controller_hid, controller_hid)

        self.linear = nn.Sequential(
            nn.Linear(self.state_dim, controller_hid),
            nn.ReLU(),
            nn.Linear(controller_hid, controller_hid),
            nn.ReLU()
        )

        self.mean_linear = nn.Linear(controller_hid, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(controller_hid, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        # self.log_std_min = -20
        # self.log_std_max = 2
        self.std_min = 1e-3
        self.std_max = 1

        # self.linear = nn.Linear(state_dim, action_dim)
        # self.linear = nn.Sequential(
        #    nn.Linear(state_dim, 16),
        #    nn.Tanh(),
        #    nn.Linear(16, action_dim)
        # )

        # if not deterministic:
        #    self.std = 0.1

        # self.deterministic = deterministic

        # nn.Linear(self.state_dim, self.action_dim)        #output a p_logits for action 1
        self.device = device
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        """
        Given states input [batch, state_dim],
        """
        state = state.to(self.device)
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = self.linear(state)

        mean = self.mean_linear(x)
        # return torch.tanh(mean), 0

        log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        std = torch.clamp(std, self.std_min, self.std_max)

        normal = Normal(mean, std)
        z = normal.rsample()
        a = torch.tanh(z)

        # compute log probability
        log_pi = normal.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)

        return a, log_pi

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
        state = state.to(self.device)
        # x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = self.linear(state)

        mean = self.mean_linear(x)
        if not behaviour_uncertainty:  # no behaviour uncertainty, mean prediction
            return torch.tanh(mean).detach()

        log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        std = torch.clamp(std, self.std_min, self.std_max)

        normal = Normal(mean, std)
        z = normal.rsample()
        a = torch.tanh(z)

        return a.detach()

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

    def pg_train(self, num_epoch, num_particle, initial_state, horizon, cost_f, model_imagine_f, w_uncertainty,
                 e_uncertainty, gamma=0.95):
        """
        initial_state : [batch, state_dim]

        """
        loss_list = []
        initial_state = initial_state.expand(num_particle, self.state_dim)

        for e in range(num_epoch):
            self.optimiser.zero_grad()
            output_matrix, action_log_prob_matrix = model_imagine_f(initial_state, self.forward, horizon, plan='pg',
                                                                    W_uncertainty=w_uncertainty,
                                                                    e_uncertainty=e_uncertainty)
            cost = cost_f(output_matrix).detach()  # [seq-1, batch, 1]

            cost = cost * torch.tensor([gamma ** (t + 1) for t in range(cost.size(0))]).unsqueeze(-1).unsqueeze(-1).to(
                self.device)

            # baseline = torch.mean(cost, dim = 0).unsqueeze(0)       #[1, batch, 1]
            # cost = cost - torch.mean(cost, dim = 0).unsqueeze(0)
            # loss = ((cost-baseline) * action_log_prob_matrix).sum(0)
            loss = cost.sum(0) * action_log_prob_matrix.sum(0)

            loss = loss.sum()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.parameters(), 1)
            nn.utils.clip_grad_norm_(self.parameters(), 1)

            self.optimiser.step()
            loss_list.append(loss.item())
            if e % 50 == 0:
                print('Epoch = {}; Policy gradient training loss = {}'.format(e, loss.item()))
        return loss_list

    def rp_train(self, num_epoch, num_particle, initial_state, horizon, cost_f, model_imagine_f, w_uncertainty,
                 e_uncertainty, gamma=0.9):
        """
        From an initial state, use mode imagination function to make prediction of the next state accordin to the action proposed by
        the controller, we fixed the horizon and compute the total reward of the trajectory, from which the gradient w.r.t policy
        parameters is taken.
        inital_state: [batch, output_dim]
        """
        loss_list = []

        initial_state = initial_state.expand(num_particle, self.state_dim)

        for e in range(num_epoch):
            self.optimiser.step()
            output_matrix, _ = model_imagine_f(initial_state, self.forward, horizon, plan='rp',
                                               W_uncertainty=w_uncertainty,
                                               e_uncertainty=e_uncertainty)  # [seq-1, batch, output], [seq-1, batch, 1]

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
            nn.utils.clip_grad_norm_(self.parameters(), 1)

            self.optimiser.step()
            loss_list.append(loss.item())
            # if e%200 == 0:
            #   print('Epoch = {}; Policy gradient training loss = {}'.format(e, loss.item()))
        return loss_list


