import gym
import math
import numpy as np
import torch
import torch.nn as nn

from torch.distributions.kl import kl_divergence
from torch.distributions import Normal




class RSSM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=4, state_size=32, device='cpu',
                 mode='LSTM'):  # action, hidden and observation dim
        super(RSSM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size

        self.mode = mode
        self.device = device

        if mode == 'RNN':
            self.transition_RNN = nn.RNNCell(input_size=hidden_size, hidden_size=hidden_size)
            # self.transition_RNN = nn.RNN(input_size = input_size, hidden_size = hidden_size)
        elif mode == 'LSTM':
            self.transition_RNN = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
            # self.transition_RNN = nn.LSTM(input_size = input_size, hidden_size = hidden_size)
        elif mode == 'GRU':
            self.transition_RNN = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
            # self.transition_RNN = nn.GRU(input_size = input_size, hidden_size = hidden_size)

        # linear layer converting the state + action
        self.state_action_layer = nn.Sequential(
            nn.Linear(self.state_size + self.input_size, self.hidden_size),
            nn.ReLU()
        )
        # prior
        # self.hidden_prior = nn.Sequential(
        #    nn.Linear(self.hidden_size, self.hidden_size),
        #    nn.ReLU()
        # )
        self.prior_mean = nn.Linear(self.hidden_size, self.state_size)
        self.prior_sigma = nn.Linear(self.hidden_size, self.state_size)
        self._min_stddev = 0.1

        # poster
        self.hidden_obs = nn.Sequential(
            nn.Linear(self.hidden_size + self.output_size, self.hidden_size),
            nn.ReLU()
        )
        self.poster = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        self.post_mean = nn.Linear(self.hidden_size, self.state_size)
        self.post_sigma = nn.Linear(self.hidden_size, self.state_size)

        # decoder
        self.state_hidden = nn.Sequential(
            nn.Linear(self.state_size + self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        # self.obs = nn.Sequential(
        #    nn.Linear(self.hidden_size, self.hidden_size),
        #    nn.ReLU()
        # )
        self.obs_mean = nn.Linear(self.hidden_size, self.output_size)
        self.obs_sigma = nn.Linear(self.hidden_size, self.output_size)

        # intial hidden encoder (take the fist observation as input and output an initial hidden state)
        self.init_h = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

        self.loss_list = []

    def prior(self, state, action, rnn_hidden, rnn_hidden_c=None):
        """
        h_t+1 = f(h_t, s_t, a_t)
        prior : p(s_t+1 | h_t+1)

        state : [batch, state_dim]
        action : [batch, action_dim]
        rnn_hidden; [batch, rnn hidden dim]
        """
        state_action = self.state_action_layer(torch.cat([state, action], dim=-1))  # [batch, hidden]
        if self.mode == 'LSTM':
            rnn_hidden, rnn_hidden_c = self.transition_RNN(state_action, (rnn_hidden, rnn_hidden_c))
        else:
            rnn_hidden = self.transition_RNN(state_action, rnn_hidden)  # [batch, hidden]
        # hidden_prior = self.hidden_prior(rnn_hidden)            #[batch, hidden]
        hidden_prior = rnn_hidden
        prior_mean = self.prior_mean(hidden_prior)  # [batch, state]
        # prior_sigma = F.softplus(self.prior_sigma(hidden_prior)) + self._min_stddev     #[batch, state]
        prior_sigma = torch.exp(self.prior_sigma(hidden_prior))
        if self.mode == 'LSTM':
            return prior_mean, prior_sigma, rnn_hidden, rnn_hidden_c
        else:
            return prior_mean, prior_sigma, rnn_hidden

    def posterior(self, rnn_hidden, obs):
        """
        posterior q(s_t | h_t, o_t)

        rnn_hidden: [batch, hidden]
        embedded_obs : [batch, output_dim]
        """
        hidden_obs = self.hidden_obs(torch.cat([rnn_hidden, obs], dim=-1))  # [batch, hidden]
        poster = self.poster(hidden_obs)  # [batch, hidden]
        poster_mean = self.post_mean(poster)  # [batch, state]
        # poster_sigma = F.softplus(self.post_sigma(poster)) + self._min_stddev     #[batch, state]
        poster_sigma = torch.exp(self.post_sigma(poster))

        return poster_mean, poster_sigma

    def obs_model(self, state, rnn_hidden):
        """
        p(o_t | s_t, h_t)
        """
        state_hidden = self.state_hidden(torch.cat([state, rnn_hidden], dim=-1))  # [batch, hidden]
        # obs = self.obs(state_hidden)            #[batch, hidden]
        obs = state_hidden
        obs_mean = self.obs_mean(obs)  # [batch, output_size]
        # obs_sigma = F.softplus(self.obs_sigma(obs))+self._min_stddev         #[batch, output_size]
        obs_sigma = torch.exp(self.obs_sigma(obs))

        return obs_mean, obs_sigma

    def forward(self, X, A, beta=1, print_output=False):
        """
        Likelihood objective function for a given trajectory (change to batched verision later)
        X: data matrix of shape [seq_length, batch, output_size]]      (we only feed one trajectory here for testing)
        A: data matrix of action [seq_length-1, batch]
        """
        assert X.size(0) == A.size(0) + 1, print('the seq length of X and A are wrong')
        kl_loss = 0  # KL divergence term
        Ell_loss = 0  # expected log likelihood term
        batch_size = X.size(1)

        if len(X.size()) != 3:
            print('The input data matrix should be the shape of [seq_length, batch_size, input_dim]')

        X = X.to(self.device)
        A = A.to(self.device)

        # container
        states = torch.zeros(A.size(0), A.size(1), self.state_size).to(self.device)  # [seq-1, batch, state]
        rnn_hiddens = torch.zeros(A.size(0), A.size(1), self.hidden_size).to(self.device)  # [seq-1, batch, hidden]

        # initialising state and rnn hidden state
        # state = torch.zeros(X.size(1), self.state_size).to(self.device)
        rnn_hidden = self.init_h(X[0]).to(self.device)  # [batch, hidden]
        if self.mode == 'LSTM':
            rnn_hidden_c = torch.zeros_like(rnn_hidden).to(self.device)  # [batch, hidden]

        # temp_prior = self.hidden_prior(rnn_hidden)      #[batch, state]
        temp_prior = rnn_hidden
        prior_mean = self.prior_mean(temp_prior)  # [batch, state]
        prior_sigma = torch.exp(self.prior_sigma(temp_prior))  # [batch, state]
        state = self.reparametrise(prior_mean, prior_sigma)  # [batch, state]

        # rnn_hidden = torch.zeros(X.size(1), self.hidden_size).to(self.device)

        # emission_mean = X[0]
        for t in range(1, X.size()[
            0]):  # for each time step, compute the free energy for each batch of data (start from the second hid state)
            if self.mode == 'LSTM':
                next_state_prior_m, next_state_prior_sigma, rnn_hidden, rnn_hidden_c = self.prior(state,
                                                                                                  A[t - 1].unsqueeze(
                                                                                                      -1),
                                                                                                  rnn_hidden,
                                                                                                  rnn_hidden_c)
            else:
                next_state_prior_m, next_state_prior_sigma, rnn_hidden = self.prior(state, A[t - 1].unsqueeze(-1),
                                                                                    rnn_hidden)

            next_state_post_m, next_state_post_sigma = self.posterior(rnn_hidden, X[t])
            state = self.reparametrise(next_state_post_m, next_state_post_sigma)  # [batch, state_size]
            states[t - 1] = state
            rnn_hiddens[t - 1] = rnn_hidden
            next_state_prior = Normal(next_state_prior_m, next_state_prior_sigma)
            next_state_post = Normal(next_state_post_m, next_state_post_sigma)

            # kl = kl_divergence(next_state_prior, next_state_post).sum(dim=1)        #[batch]
            kl = kl_divergence(next_state_post, next_state_prior).sum(dim=1)  # [batch]

            kl_loss += kl.mean()
        kl_loss /= A.size(0)

        # compute nll

        # flatten state
        flatten_states = states.view(-1, self.state_size)
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.hidden_size)
        flatten_x_mean, flatten_x_sigma = self.obs_model(flatten_states, flatten_rnn_hiddens)

        nll = self.batched_gaussian_ll(flatten_x_mean, flatten_x_sigma, X[1:, :, :].reshape(-1, self.output_size))
        nll = nll.mean()

        FE = nll - kl_loss

        if print_output:
            # print('ELL loss=', Ell_loss, 'KL loss=', kl_loss)
            print('Free energy of this batch = {}. Nll loss = {}. KL div = {}.'.format(float(FE.data)
                                                                                       , float(nll.data),
                                                                                       float(kl_loss.data)))

        return FE, nll, kl_loss

    def mc_predict(self, initial_obs, actions, mean_obs=False):
        """
        initial_obs : [1, output_dim]
        actions: [seq-1, 1, action_dim]
        """

        initial_obs = initial_obs.to(self.device)
        actions = actions.to(self.device)

        total_list = []

        time_step = actions.size(0)

        total_list = []
        # [1, output]
        for i in range(200):
            temp_pred = []
            # container
            # states = torch.zeros(actions.size(0), actions.size(1), self.state_size).to(self.device)         #[seq-1, 1, state]
            # rnn_hiddens = torch.zeros(actions.size(0), actions.size(1), self.hidden_size).to(self.device)   #[seq-1, 1, hidden]

            # initialising state and rnn hidden state
            # state = torch.zeros(initial_obs.size(0), self.state_size).to(self.device)             #[1, state]
            rnn_hidden = self.init_h(initial_obs)  # [1, hidden]
            if self.mode == 'LSTM':
                rnn_hidden_c = torch.zeros_like(rnn_hidden)  # [1, hidden]

            # temp_prior = self.hidden_prior(rnn_hidden)      #[1, state]
            temp_prior = rnn_hidden
            prior_mean = self.prior_mean(temp_prior)
            prior_sigma = torch.exp(self.prior_sigma(temp_prior))
            state = self.reparametrise(prior_mean, prior_sigma)

            # x_sample = initial_obs      #[1, output_dim]
            for t in range(time_step):
                if self.mode == 'LSTM':
                    next_state_prior_m, next_state_prior_sigma, rnn_hidden, rnn_hidden_c = self.prior(state,
                                                                                                      actions[t],
                                                                                                      rnn_hidden,
                                                                                                      rnn_hidden_c)
                else:
                    next_state_prior_m, next_state_prior_sigma, rnn_hidden = self.prior(state, actions[t], rnn_hidden)

                state = self.reparametrise(next_state_prior_m, next_state_prior_sigma)

                # next_state_post_m, next_state_post_sigma = self.posterior(rnn_hidden, x_sample)
                # state = self.reparametrise(next_state_post_m, next_state_post_sigma)        #[batch, state_size]

                x_mean, x_sigma = self.obs_model(state, rnn_hidden)
                if mean_obs:
                    x_sample = x_mean
                else:
                    x_sample = self.reparametrise(x_mean, x_sigma)  # [1, output_dim]

                temp_pred.append(x_sample.unsqueeze(0))  # list of shape [1,1,output]
            temp_pred_vec = torch.cat(temp_pred, dim=0)  # [seq-1, 1, output]

            total_list.append(temp_pred_vec.unsqueeze(-1))  # list of shape [seq-1, 1, output, 1]
        total_list = torch.cat(total_list, dim=-1)  # [seq-1, 1, output, 200]
        mean = total_list.mean(dim=-1)  # [seq-1, 1, output]
        std = total_list.std(dim=-1)  # [seq-1, 1, output]

        return mean, std

    def imagine(self, init_x, control_f, horizon, plan, mean_obs=False):
        """
        init_x : [batch, output]
        """
        init_x = init_x.to(self.device)
        rnn_hidden = self.init_h(init_x)
        if self.mode == 'LSTM':
            rnn_hidden_c = torch.zeros_like(rnn_hidden).to(self.device)

        # temp_prior = self.hidden_prior(rnn_hidden)      #[1, state]
        temp_prior = rnn_hidden
        prior_mean = self.prior_mean(temp_prior)
        prior_sigma = torch.exp(self.prior_sigma(temp_prior))
        state = self.reparametrise(prior_mean, prior_sigma)

        x_sample = init_x
        pred = []
        action_log_prob_list = []
        for t in range(horizon):
            if plan == 'pg':
                action_samples, action_log_prob = control_f(x_sample)
                action_log_prob_list.append(action_log_prob.unsqueeze(0))  # [1, batch, 1]

            elif plan == 'rp':
                action_samples, _ = control_f(x_sample)  # [batch, 1]
                action_log_prob_list = 0
            else:
                raise NotImplementedError

            if self.mode == 'LSTM':
                next_state_prior_m, next_state_prior_sigma, rnn_hidden, rnn_hidden_c = self.prior(state,
                                                                                                  action_samples,
                                                                                                  rnn_hidden,
                                                                                                  rnn_hidden_c)
            else:
                next_state_prior_m, next_state_prior_sigma, rnn_hidden = self.prior(state, action_samples, rnn_hidden)

            state = self.reparametrise(next_state_prior_m, next_state_prior_sigma)

            x_mean, x_sigma = self.obs_model(state, rnn_hidden)
            if mean_obs:
                x_sample = x_mean
            else:
                x_sample = self.reparametrise(x_mean, x_sigma)  # [1, output_dim]
            pred.append(x_sample.unsqueeze(0))

        if plan == 'pg':
            action_log_prob_list = torch.cat(action_log_prob_list)  # [seq-1, batch, 1]

        return torch.cat(pred), action_log_prob_list

    def validate_by_imagination(self, init_x, control_f, plan, mean_obs=False):
        """
        Perform planning on learnt model as opposed to real dynamics
        """
        init_x = init_x.to(self.device)
        rnn_hidden = self.init_h(init_x)
        if self.mode == 'LSTM':
            rnn_hidden_c = torch.zeros_like(rnn_hidden).to(self.device)

        # temp_prior = self.hidden_prior(rnn_hidden)      #[1, state]
        temp_prior = rnn_hidden
        prior_mean = self.prior_mean(temp_prior)
        prior_sigma = torch.exp(self.prior_sigma(temp_prior))
        state = self.reparametrise(prior_mean, prior_sigma)

        x_sample = init_x
        pred = []
        action_log_prob_list = []
        reward = 0
        iter = 0

        while True:
            action_samples, _ = control_f(x_sample)  # [batch, 1]

            if self.mode == 'LSTM':
                next_state_prior_m, next_state_prior_sigma, rnn_hidden, rnn_hidden_c = self.prior(state, action_samples,
                                                                                                  rnn_hidden,
                                                                                                  rnn_hidden_c)
            else:
                next_state_prior_m, next_state_prior_sigma, rnn_hidden = self.prior(state, action_samples, rnn_hidden)

            state = self.reparametrise(next_state_prior_m, next_state_prior_sigma)

            x_mean, x_sigma = self.obs_model(state, rnn_hidden)
            if mean_obs:
                x_sample = x_mean
            else:
                x_sample = self.reparametrise(x_mean, x_sigma)  # [1, output_dim]

            reward += 1
            iter += 1

            done = x_sample[:, 0] < -2.4 \
                   or x_sample[:, 0] > 2.4 \
                   or x_sample[:, 2] < -12 * 2 * math.pi / 360 \
                   or x_sample[:, 2] > 12 * 2 * math.pi / 360 \
                   or iter >= 200
            done = bool(done)
            if done:
                break

        return reward

    def reparametrise(self, mean, sigma):
        """
        sigma should have the same shape as mean (no correaltion)
        """
        eps = torch.rand_like(sigma).normal_()
        eps = eps.to(self.device)
        return mean + sigma * eps

    def batched_gaussian_ll(self, mean, sigma, x):
        """
        log-likelihood of batched observation
        mean : shape [batch, output_size]
        sigma  : shape [batch, output_size]   (diagonal covariance)
        x    : shape [batch, output_size]
        the shape of final result is [batch, ]
        """
        # mean = mean.to(self.device)
        # sigma = sigma.to(self.device)
        if 0 in sigma:
            # sigma = sigma + 1e-10
            print('Zero occurs in diagonal sigma matrix. (batched gaussian ll)')
        if 0 in sigma ** 2:
            print('Zero occurs after squaring sigma matrix. (batched gaussian ll)')

        inv_diag_cov = self.diagonalise(1 / (sigma ** 2),
                                        batch=True)  # a 2d batched matrix----> 3d batched diagonal tensor

        exp = ((x - mean).unsqueeze(-2)) @ inv_diag_cov @ ((x - mean).unsqueeze(-1))  #
        exp = exp.squeeze()  # [batch]
        # print(exp)

        if 0 in torch.prod(sigma ** 2, dim=-1):
            print('Zero occurs when calculating determinant of diagonal covariance. (batched gaussian ll)')

        logdet = torch.sum(2 * torch.log(sigma), dim=-1)
        # logdet = torch.log(torch.prod(sigma**2, dim = -1))         #product of all diagonal variance for each batch, shape [batch]
        # print('logdet=', logdet)
        n = mean.size()[-1]

        return -(n / 2) * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * exp  # need double checking

    def diagonalise(self, input, batch):
        """
        if input a vector, return a diagonal matrix
        if input a non-batched 2d matrix, return a diagonal matrix, eg: [[1,2],[3,4]] ---> diag([1,2,3,4])
        if input a batched 2d matrix, return a batched diagonal matrix
        if input a 3d batched tensor, return a batched diagonal tensor
        """
        if len(input.size()) == 1:
            return torch.diag(input)
        if len(input.size()) == 2:
            if not batch:
                return torch.diag(vec(input))
            else:
                bdiag = torch.Tensor().to(self.device)
                for i in range(input.size()[0]):
                    bdiag = torch.cat((bdiag, torch.diag(input[i]).unsqueeze(0)), axis=0)
                return bdiag

        if len(input.size()) == 3 and batch:
            bdiag = torch.Tensor()
            for i in range(input.size()[0]):
                bdiag = torch.cat((bdiag, torch.diag(vec(input[i])).unsqueeze(0)), axis=0)

            return bdiag
        else:
            print('Dimension of inpout tensor should only be 1,2,3.')

    def print_loss(self):
        return self.loss_list

    def print_params(self):

        pass

