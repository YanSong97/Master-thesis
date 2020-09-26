
import torch
import torch.nn as nn


class controller(nn.Module):
    def __init__(self, action_dim=1, state_dim=4, deterministic=True, device='cuda'):
        super(controller, self).__init__()
        self.action_dim = action_dim
        controller_hid = 16
        self.state_dim = state_dim
        init_w = 1e-3

        self.linear = nn.Sequential(
            nn.Linear(self.state_dim, controller_hid),
            nn.ReLU(),
            nn.Linear(controller_hid, controller_hid),
            nn.ReLU(),
            nn.Linear(controller_hid, action_dim)
        )

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
        a = self.linear(state)
        a = torch.tanh(a)
        log_pi = 0

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
        a = self.linear(state)
        a = torch.tanh(a)

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

    def pg_train(self, num_epoch, initial_state, horizon, cost_f, model_imagine_f, w_uncertainty, e_uncertainty,
                 gamma=0.95):
        """
        initial_state : [batch, state_dim]

        """
        loss_list = []
        num_particle = 100
        initial_state = initial_state.expand(num_particle, self.state_dim)

        for e in range(num_epoch):
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
            nn.utils.clip_grad_norm_(self.parameters(), 5)

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
        num_particle = num_particle
        initial_state = initial_state.expand(num_particle, self.state_dim)

        cost_mean_list = []
        cost_std_list = []

        for e in range(num_epoch):
            self.optimiser.zero_grad()
            output_matrix, action_matrix = model_imagine_f(initial_state, self.forward, horizon, plan='rp',
                                                           W_uncertainty=w_uncertainty, e_uncertainty=e_uncertainty)

            self.action_matrix = action_matrix
            self.temp_output_matrix = torch.cat([initial_state.unsqueeze(0).to(self.device), output_matrix], dim=0)

            cost = cost_f(output_matrix)  # [seq-1, batch, 1]

            mean_cost = cost.data.sum(0).mean(0)  # []
            std_cost = cost.data.sum(0).std(0)
            cost_mean_list.append(mean_cost)
            cost_std_list.append(std_cost)

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
            # print('policy loss = {}', loss.item())
            # print('Epoch = {}; Policy gradient training loss = {}; Cost: mean {} std {}.'.format(e, loss.item()/num_particle,
            #                                                                                        mean_cost.item(), std_cost.item()))

        return loss_list, torch.cat(cost_mean_list), torch.cat(cost_std_list)

    def rp_validate(self, num_particle, initial_state, horizon, cost_f, model_imagine_f, w_uncertainty, e_uncertainty,
                    gamma=1):
        initial_state = initial_state.expand(num_particle, self.state_dim)
        output_matrix, action_matrix = model_imagine_f(initial_state, self.forward, horizon, plan='rp',
                                                       W_uncertainty=w_uncertainty, e_uncertainty=e_uncertainty)
        cost = cost_f(output_matrix)

        mean_cost = cost.data.sum(0).mean(0)
        return mean_cost.item()
