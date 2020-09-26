import torch
import numpy as np
import math
import torch.nn as nn



class SRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, mode, noise=None):
        super(SRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.mode = mode

        if mode == 'RNN':
            self.transition = nn.RNN(input_dim, hidden_dim)
        elif mode == 'LSTM':
            self.transition = nn.LSTM(input_dim, hidden_dim)
        elif mode == 'GRU':
            self.transition = nn.GRU(input_dim, hidden_dim)
        else:
            raise ValueError('Mode should be one of RNN, LSTM or GRU')

        self.init_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.emission_mean = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

        if noise is None:
            self.emission_sigma = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
            self.noise = None
        elif noise == 'Param':
            logvar = torch.tensor([1., 1., 1., 1.])
            self.emission_logvar = nn.Parameter(logvar.clone().requires_grad_(True))
            self.noise = 'Param'

        else:
            self.noise = noise.to(self.device)

        def init_normal(m):
            if type(m) == nn.Linear:
                #    nn.init.uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # self.emission_sigma.apply(init_normal)

    def forward(self, init_X, A):
        """
        init_X : [batch, output_dim]
        A : [seq, batch, input_dim]
        """
        init_X = init_X.to(self.device)
        A = A.to(self.device)

        init_h = self.init_encoder(init_X).unsqueeze(0).to(self.device)  # [batch, hidden]

        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)
            recurrent_state = self.transition(A, (init_h, init_c))[0]
        else:
            recurrent_state = self.transition(A, init_h)[0]

        pred_obs_list = []
        mean_list = []
        sigma_list = []  # h of shape [batch, hidden]

        for h in recurrent_state:

            emission = self.emission(h)  # [batch, hidden]
            emission_mean = self.emission_mean(emission)  # [batch, out]

            if self.noise is None:
                emission_sigma = torch.exp(self.emission_sigma(emission))  # [batch, out]
            elif self.noise == 'Param':
                emission_sigma = torch.exp(self.emission_logvar.expand(emission_mean.shape))
            else:
                emission_sigma = self.noise.unsqueeze(0).expand(emission_mean.shape)  # [batch, out]

            x_sample = self.reparametrise(emission_mean, emission_sigma)  # [batch, output]
            mean_list.append(emission_mean)
            sigma_list.append(emission_sigma)
            pred_obs_list.append(x_sample.unsqueeze(0))

        return pred_obs_list, mean_list, sigma_list

    def mc_predict(self, init_X, A):
        """
        X : [batch, output_dim]
        A : [seq-1, batch, action_dim]
        """
        init_X = init_X.to(self.device)
        A = A.to(self.device)
        total_list = []
        init_h = self.init_encoder(init_X).unsqueeze(0).to(self.device)  # [1, batch, hidden]
        for i in range(200):
            temp_pred = []

            if self.mode == "LSTM":
                init_c = torch.zeros_like(init_h)
                recurrent_state = self.transition(A, (init_h, init_c))[0]
            else:
                recurrent_state = self.transition(A, init_h)[0]

            t = 1
            for h in recurrent_state:
                emission = self.emission(h)
                emission_mean = self.emission_mean(emission)  # [batch, output_dim]
                if self.noise is None:
                    emission_sigma = torch.exp(self.emission_sigma(emission))  # [batch, output_dim]
                elif self.noise == 'Param':
                    emission_sigma = torch.exp(self.emission_logvar.expand(emission_mean.shape))
                else:
                    emission_sigma = self.noise.unsqueeze(0).expand(emission_mean.shape)  # [batch, output]

                # sample
                x_sample = self.reparametrise(emission_mean, emission_sigma).unsqueeze(0)  # [1, batch,output_dim]

                temp_pred.append(x_sample)
                t += 1
            temp_pred_vec = torch.cat(temp_pred)  # [seq-1, batch,output_dim]

            total_list.append(temp_pred_vec.unsqueeze(-1))  # [seq-1, batch, output_dim, 1]
        total_list = torch.cat(total_list, dim=-1)  # [seq-1, batch, output_dim, 500]

        mean = torch.mean(total_list, dim=-1)  # [seq-1, batch, output_dim]
        std = torch.std(total_list, dim=-1)  # [seq-1, batch, output_dim]

        return mean, std

    def forward_likelihood(self, X, A):
        """
        X : [seq, batch, output_dim]
        A : [seq-1, batch, intput_dim]
        """
        init_h = self.init_encoder(X[0, :, :])  # [batch, hidden]
        recurrent_state = self.transition(A, init_h.unsqueeze(0))[0]

        pred_obs_list = []
        mean_list = []
        sigma_list = []
        ll = 0
        t = 1
        for h in recurrent_state:
            emission = self.emission(h)  # [batch, hidden]
            emission_mean = self.emission_mean(emission)  # [batch, out]
            emission_sigma = torch.abs(self.emission_sigma(emission))  # [batch, out]
            # compute the log-likelihood
            temp_ll = self.batched_gaussian_ll(emission_mean, emission_sigma, X[t])
            ll += temp_ll
            t += 1
        return ll.sum()

    def batched_gaussian_ll(self, mean, sigma, x):
        """
        log-likelihood of batched observation
        mean : shape [batch, output_size]
        sigma  : shape [batch, output_size]   (diagonal covariance)
        x    : shape [batch, output_size]
        the shape of final result is [batch, ]
        """
        if 0 in sigma:
            print('Zero occurs in diagonal sigma matrix. (batched gaussian ll)')
        if 0 in sigma ** 2:
            print('Zero occurs after squaring sigma matrix. (batched gaussian ll)')

        inv_diag_cov = diagonalise(1 / (sigma ** 2), batch=True,
                                   device=self.device)  # a 2d batched matrix----> 3d batched diagonal tensor

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

    def reparametrise(self, mean, sigma):
        """
        sigma should have the same shape as mean (no correaltion)
        """
        eps = torch.rand_like(sigma).normal_().to(self.device)

        return mean + sigma * eps

    def imagine(self, init_x, control_f, horizon, plan):
        """
        Given initial observations,
        init_x : [batch, output_dim]
        """
        init_x = init_x.to(self.device)
        init_h = self.init_encoder(init_x).unsqueeze(0)  # [1, batch, hidden]

        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)
            previous_c = init_c

        previous_x = init_x
        previous_h = init_h
        output_list = []
        action_log_prob_list = []

        for t in range(horizon):
            # action
            if plan == 'pg':
                action_dist = control_f(previous_x)  # [batch, 1]
                # try deterministic policy
                # action_samples = (action_dist.probs > 0.5).float()
                # action_log_prob = action_dist.log_prob(action_samples)
                # action_log_prob_list.append(action_log_prob.unsqueeze(0))

                action_samples = action_dist.sample()  # [batch, 1]
                # compute log prob
                action_log_prob = action_dist.log_prob(action_samples)  # [batch, 1]
                action_log_prob_list.append(action_log_prob.unsqueeze(0))  # [1, batch, 1]
            elif plan == 'rp':
                action_samples, _ = control_f(previous_x)  # [batch, 1]
                action_log_prob_list = 0

            if self.mode == 'LSTM':
                next_h, next_c = self.transition(action_samples.unsqueeze(0), (previous_h, previous_c))[1]
                previous_c = next_c
            else:
                next_h = self.transition(action_samples.unsqueeze(0), previous_h)[1]  # [1, batch, hidden]

            emission = self.emission(next_h.squeeze(0))  # [batch, hid]
            emission_mean = self.emission_mean(emission)  # [batch, output]

            if self.noise is None:
                emission_sigma = torch.exp(self.emission_sigma(emission))  # [batch, out]
            elif self.noise == 'Param':
                emission_sigma = torch.exp(self.emission_logvar.expand(emission_mean.shape))
            else:

                emission_sigma = self.noise.unsqueeze(0).expand(emission_mean.shape)  # [batch, out]

            next_x = self.reparametrise(emission_mean, emission_sigma)  # [batch, output]
            output_list.append(next_x.unsqueeze(0))  # [1, batch, output]

            previous_x = next_x
            previous_h = next_h

        output_list = torch.cat(output_list)  # [seq-1, batch, output]
        if plan == 'pg':
            action_log_prob_list = torch.cat(action_log_prob_list)  # [seq-1, batch, 1]

        return output_list, action_log_prob_list

    def validate_by_imagination(self, init_x, control_f, plan):

        init_x = init_x.to(self.device)
        init_h = self.init_encoder(init_x).unsqueeze(0)  # [1, batch, hidden]
        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)
            previous_c = init_c
        previous_x = init_x
        previous_h = init_h

        reward = 0
        iter = 0
        while True:
            action_samples, _ = control_f(previous_x)  # [batch, 1]

            if self.mode == 'LSTM':
                next_h, next_c = self.transition(action_samples.unsqueeze(0), (previous_h, previous_c))[1]
                previous_c = next_c
            else:
                next_h = self.transition(action_samples.unsqueeze(0), previous_h)[1]  # [1, batch, hidden]

            emission = self.emission(next_h.squeeze(0))  # [batch, hid]
            emission_mean = self.emission_mean(emission)
            if self.noise is None:
                emission_sigma = torch.exp(self.emission_sigma(emission))  # [batch, out]
            elif self.noise == 'Param':
                emission_sigma = torch.exp(self.emission_logvar.expand(emission_mean.shape))
            else:

                emission_sigma = self.noise.unsqueeze(0).expand(emission_mean.shape)  # [batch, out]

            next_x = self.reparametrise(emission_mean, emission_sigma)  # [1, output]

            reward += 1
            iter += 1

            done = next_x[:, 0] < -2.4 \
                   or next_x[:, 0] > 2.4 \
                   or next_x[:, 2] < -12 * 2 * math.pi / 360 \
                   or next_x[:, 2] > 12 * 2 * math.pi / 360 \
                   or iter >= 200
            done = bool(done)

            if done:
                break

            previous_x = next_x
            previous_h = next_h

        return reward

