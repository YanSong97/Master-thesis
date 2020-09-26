import torch

import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions import Bernoulli



class controller(nn.Module):
    def __init__(self, action_dim=1, state_dim=4, device='cuda'):
        super(controller, self).__init__()
        self.action_dim = action_dim
        # self.controller_hid = controller_hid
        self.state_dim = state_dim

        # self.linear1 = nn.Linear(self.state_dim, controller_hid)
        # self.linear2 = nn.Linear(controller_hid, self.action_dim)
        self.linear = nn.Sequential(nn.Linear(state_dim, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, action_dim))
        # self.linear = nn.Linear(state_dim, action_dim)

        # nn.Linear(self.state_dim, aself.action_dim)        #output a p_logits for action 1
        self.device = device
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        """
        Given states input [4,], output an Bernoulli action distribution
        """
        state = state.to(self.device)
        p_logits = self.linear(state)  # [1,]
        p = torch.sigmoid(p_logits)
        action_dist = Bernoulli(probs=p)
        return action_dist

    def make_decision(self, state, behaviour_uncertainty, grad=None):
        """
        given a state [4,], output a sampled action
        """

        state = state.to(self.device)
        action_dist = self.forward(state)
        # action = action_dist.sample()
        if behaviour_uncertainty:
            action = action_dist.sample()
            return action.detach()

        else:
            return (action_dist.probs > 0.5).float()
            # if action_dist.probs > 0.5:
            #    return torch.tensor([1.])
            # else:
            #    return torch.tensor([0.])

    def gumbel_sample(self, s):
        """
        Reparametrisation of Bernoulli distribution
        s : [batch, output_dim]
        """
        # s=F.relu(self.linear1(s))
        # p_logits=self.linear2(s)
        p_logits = self.linear(s)  # [batch, 1]
        action_dis = RelaxedBernoulli(temperature=0.8, logits=p_logits)
        action = action_dis.rsample()
        hard_action = (action > 0.5).float()
        return action + (hard_action - action).detach()  #

    def rp_train(self, num_epoch, initial_state, horizon, cost_f, model_imagine_f, gamma=1):
        """
        Reparametrisation of Bernoulli distribution
        """

        loss_list = []
        # num_particles = 30
        # initial_state = initial_state.expand(num_particles, self.state_dim)
        for e in range(1, num_epoch + 1):
            self.optimiser.zero_grad()
            # feed initial state and the policy function to the model
            output_matrix, _ = model_imagine_f(initial_state, self.gumbel_sample, horizon, 'rp',
                                               e_uncertainty=False)  # [seq, batch, output_dim]

            cost = cost_f(output_matrix)  # [seq, batch, 1]
            cost = cost * torch.tensor([gamma ** (t + 1) for t in range(cost.size(0))]).unsqueeze(-1).unsqueeze(-1).to(
                self.device)
            cost = cost.sum()

            cost.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimiser.step()
            loss_list.append(cost.item())
            if e % 100 == 0:
                print('Epoch = {}, loss of RP policy gradient = {}'.format(e, cost.item()))

        return loss_list

    def pg_train(self, num_epoch, initial_state, horizon, cost_f, model_imagine_f, gamma=1):
        """
        From an initial state, use mode imagination function to make prediction of the next state accordin to the action proposed by
        the controller, we fixed the horizon and compute the total reward of the trajectory, from which the gradient w.r.t policy
        parameters is taken.
        inital_state: [1, output_dim]
        """
        loss_list = []
        # num_particles = 50
        # print('initial state dim', initial_state.shape)
        # initial_state = initial_state.expand(num_particles, self.state_dim)         #[num, state]

        for e in range(num_epoch):

            self.optimiser.zero_grad()

            output_matrix, action_log_prob_matrix = model_imagine_f(initial_state, self.forward, horizon, 'pg',
                                                                    e_uncertainty=False)  # [seq-1, batch, output], [seq-1, batch, 1]
            cost = cost_f(output_matrix).detach()  # [seq-1, batch, 1]

            # multiply by discount factor
            # cost = cost *  ((torch.arange(cost.size(0)+1,1,-1).float()).unsqueeze(-1).unsqueeze(-1)/cost.size(0)
            #                    ).expand(cost.shape).float().to(self.device)

            cost = cost * torch.tensor([gamma ** (t + 1) for t in range(cost.size(0))]).unsqueeze(-1).unsqueeze(-1).to(
                self.device)
            # baseline = torch.mean(cost, dim = 0).unsqueeze(0)       #[1, batch, 1]
            loss = cost.sum(0) * action_log_prob_matrix.sum(0)

            # loss = ((cost-baseline) * action_log_prob_matrix).sum(0)
            # loss = cost.sum(0)      #[batch, 1]
            # loss = torch.exp(action_log_prob_matrix.sum(0)) * cost.sum(0)
            # loss = (cost * action_log_prob_matrix).sum(0)                 #[batch, 1]
            # loss = cost.sum(0) * action_log_prob_matrix.sum(0)
            # loss = torch.mean(cost.sum(0))
            loss = loss.sum()
            loss.backward()

            # nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.optimiser.step()
            loss_list.append(loss.item())
            if e % 100 == 0:
                print('Epoch = {}; Policy gradient training loss = {}'.format(e, loss.item()))
        return loss_list


