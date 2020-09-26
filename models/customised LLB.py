

import math
import numpy as np

import torch
import torch.nn as nn

from torch.distributions.kl import kl_divergence as KL_f


from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence



class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))


        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy, outgate


class BRNN(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim, device, mode='LSTM'):
        super(BRNN, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.mode = mode

        init_dim = hidden_dim
        emission_dim = hidden_dim

        self.initial_encoder = nn.Sequential(
            nn.Linear(output_dim, init_dim),
            nn.ReLU(),
            nn.Linear(init_dim, hidden_dim),
            nn.Tanh()
        )

        self.recurrent_cell = LSTMCell(action_dim, hidden_dim)

        # self.W_mu = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim + 1).to(self.device), requires_grad=True)
        # self.W_logvar = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim + 1).to(self.device), requires_grad=True)

        self.prior_W_mu = torch.zeros_like(torch.Tensor(hidden_dim, hidden_dim + 1)).to(self.device)
        self.prior_W_mu = nn.init.kaiming_uniform_(self.prior_W_mu, a=math.sqrt(5))

        self.prior_W_logvar = torch.ones_like(self.prior_W_mu).to(self.device)  # log var
        self.prior_W_logvar = (np.log(0.05) * self.prior_W_logvar).requires_grad_(False)

        self.prior_W_logvar.data.uniform_(np.log(0.05), np.log(0.1))
        # self.prior_W_logvar.data.uniform_(np.log(0.5), np.log(0.9))

        self.W_mu = nn.Parameter(self.prior_W_mu.detach().clone().requires_grad_(True))
        self.W_logvar = nn.Parameter(self.prior_W_logvar.detach().clone().requires_grad_(True))  # log(sigma)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, emission_dim),
            nn.ReLU()
            # nn.Linear(emission_dim, output_dim)
        )

        self.emission_mean = nn.Linear(hidden_dim, output_dim)

        self.emission_logvar = nn.Linear(hidden_dim, output_dim)

        # self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.W_mu.size(1))
        logvar_init = math.log(stdv) * 2
        self.W_mu.data.uniform_(-stdv, stdv)
        self.W_logvar.data.fill_(math.log(0.05))

        # nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        # self.W_sigma.data.fill_(np.log(0.5))

    def rollout(self, init_x, A, W_eps=None, W_uncertainty=True, epsilon_uncertainty=True, track_o=False):
        """
        take initial obs and sequence of actions, output predicted observation mean and variance,
        if grad = None, using mean, if not , use posterior sharpening

        init_x : [batch, output_dim]
        A : [seq-1, batch, 1]
        return : observational mean and sigma
        """
        if track_o:
            o_list = []

        # forward propagation
        init_x = init_x.to(self.device)
        A = A.to(self.device)

        batch_size = init_x.size(0)

        previous_h = self.initial_encoder(init_x).unsqueeze(0).to(self.device)  # [1, batch ,hidden]
        if self.mode == 'LSTM':
            previous_c = torch.zeros_like(previous_h).to(self.device)

        # here we follow the BBB paper where they fix the weight for each mini-batch instead of conditioning on single example
        # if W_eps is not None:
        #    std = torch.exp(0.5 * self.W_logvar)
        #    if len(W_eps.shape) > 2:  # Multiple specified samples from W
        #        W = self.W_mu.unsqueeze(0).expand(W_eps.shape) + W_eps*std.unsqueeze(0).expand(W_eps.shape)  #  [batch, hid, hid+1]

        #    else:
        #        W = self.W_mu + W_eps*std

        # else:
        #    W_eps = torch.rand_like(self.W_logvar).normal_().to(self.device)
        #    W = self.W_mu + W_eps*torch.exp(0.5*self.W_logvar)

        # eps = torch.FloatTensor(batch_size, self.W_mu.size(0), self.W_mu.size(1)).normal_().to(self.device)
        # W =  self.W_mu.unsqueeze(0).expand(eps.shape) + eps*torch.exp(0.5*self.W_logvar).unsqueeze(0).expand(eps.shape)  # [batch, hid, hid+1]
        if W_uncertainty:
            W = self.stack_W(batch_size, self.W_mu, torch.exp(0.5 * self.W_logvar))  # [batch, hid, hid+1]
        else:
            W = self.W_mu.unsqueeze(0).expand(batch_size, self.W_mu.size(0), self.W_mu.size(1))
        # W_sample = self.reparametrise(self.W_mu, torch.exp(self.W_sigma))
        # W = W_sample.unsqueeze(0).expand(batch_size, W_sample.size(0), W_sample.size(1))        #[batch, hid, hid+1]

        output_mean_list = []
        output_std_list = []
        preds = []
        for t in range(A.size(0)):
            previous_a = A[t].unsqueeze(0)  # [1, batch, 1]
            if self.mode == 'LSTM':
                # current_h, current_c = self.transition(previous_a, (previous_h, previous_c))[-1]   #[1,batch, hidden]
                current_h, previous_c, o_t = self.recurrent_cell(previous_a.squeeze(0),
                                                                 (previous_h.squeeze(0), previous_c.squeeze(0)))
                current_h = current_h.unsqueeze(0)
                previous_c = previous_c.unsqueeze(0)
                if track_o:
                    o_list.append(o_t)  # [batch, hidden]

            else:
                current_h = self.transition(previous_a, previous_h)[-1]  # [1, batch, hidden]

            f_t = torch.cat([current_h, torch.ones(current_h.shape[0], current_h.shape[1], 1).to(self.device)], dim=-1)
            f_t = f_t.permute(1, 2, 0)  # [batch, hid+1, 1]

            # one = torch.ones(current_h.squeeze(0).size()[0]).to(self.device)
            # f_t = torch.cat((current_h.squeeze(0), one.unsqueeze(1)), dim = 1)           #[batch, hidden_size + 1]
            # next_h = torch.bmm(W, f_t.unsqueeze(-1)).squeeze(-1)         #[batch,hid, hid+1] *[batch, hid+1, 1] ----> [batch, hid]
            next_h = torch.bmm(W, f_t).squeeze(-1)

            temp_emission = self.decoder(next_h)
            emission_mean = self.emission_mean(temp_emission)  # [batch, output]
            emission_var = torch.exp(self.emission_logvar(temp_emission))
            emission_sigma = torch.sqrt(emission_var)
            # emission_sigma = torch.sqrt(self.x_var.expand(emission_mean.shape))

            if epsilon_uncertainty:
                emission = self.reparametrise(emission_mean, emission_sigma)  # [batch, output]
            else:
                emission = emission_mean  # [batch, output]

            preds.append(emission.unsqueeze(0))

            # emission_mean = self.decoder_mean(emission)         #[batch, output]
            # emission_sigma = torch.exp(self.decoder_sigma(emission))       #[batch, output]
            # if self.mode == 'LSTM':
            #    previous_c = current_c
            previous_h = next_h.unsqueeze(0)  # !!!!!!!!!!  [1, batch, hid]

            output_mean_list.append(emission_mean.unsqueeze(0))
            output_std_list.append(emission_sigma.unsqueeze(0))
        output_mean_list = torch.cat(output_mean_list, dim=0)  # [seq-1, batch, output]
        output_std_list = torch.cat(output_std_list, dim=0)  # [seq-1, batch, output]

        if track_o:
            return torch.cat(preds, dim=0), output_mean_list, output_std_list, torch.stack(o_list)

        return torch.cat(preds, dim=0), output_mean_list, output_std_list

    def forward(self, X, A, N):
        """
        X : [seq, batch, output_dim]
        A : [seq-1, batch, 1]
        """
        X = X.to(self.device)
        A = A.to(self.device)
        # compute nll
        # output_mean, output_std = self.rollout(X[0], A)
        # NLL = self.get_nll(output_mean, output_std, X[1:, :, :])        #[batch]
        # The gradient of nll
        # fix_eps = torch.FloatTensor(X.size(1), self.W_logvar.shape[0], self.W_logvar.shape[1]).normal_().to(self.device)

        # forward pass with sharpening posterior weight
        _, output_mean, output_cov, o_matrix = self.rollout(X[0], A, track_o=True)
        self.o_matrix = o_matrix

        return self.get_loss(output_mean, output_cov, X[1:], N)

    def get_loss(self, output_mean, output_sigma, target, N):
        """
        calculate free energy, NLL - KL
        """
        # NLL = self.get_nll(output_mean, output_sigma, target)
        # T = target.size(0)*target.size(1)

        # LL = 0.5 * (- T*np.log(2*np.pi) - 2*torch.log(output_sigma.sum())  -  torch.pow(output_sigma, -2).mul(torch.pow(target - output_mean, 2)).sum())

        # KL
        # KL = self.kl_divergence(self.prior_W_mu, torch.exp(self.prior_W_sigma), self.W_mu, torch.exp(self.W_sigma))

        # KL = 0.5 * ( self.prior_W_logvar - self.W_logvar - 1 \
        #            + torch.exp(self.W_logvar - self.prior_W_logvar) \
        #            + torch.exp(-self.prior_W_logvar) * torch.pow(self.W_mu - self.prior_W_mu, 2) ).sum()

        # FE = (1/T)*LL - (1/N)*KL

        # KL_sharp = torch.sum((self.sharp_W - self.kl_W).pow(2)/ (2 * 0.002**2))
        # KL_sharp = self.kl_divergence(self.phi_container, self.condition_prior_W_sigma, self.sharp_W_mean, self.sharp_W_sigma)

        # return FE.flatten(), (1/T)*LL.flatten(), (1/N)*KL.flatten()

        batch_size = target.size(1)
        seq_length = target.size(0)
        T = batch_size * seq_length
        # LL
        flatten_x_mean = output_mean.view(-1, self.output_dim)
        flatten_x_std = torch.sqrt(output_sigma).view(-1, self.output_dim)

        flatten_target = target.reshape(-1, self.output_dim)

        LL = self.batched_gaussian_ll(flatten_x_mean, flatten_x_std, flatten_target)
        LL = LL.sum()

        # LL = 0.5 * (- T*np.log(2*np.pi) - torch.log(output_sigma).sum()
        #        -  torch.pow(output_sigma, -1).mul(torch.pow(target - output_mean, 2)).sum())
        # print('LL = {}; LL2 = {}'.format(LL.item(), LL2.item()))
        # KL
        Wprior = Normal(self.prior_W_mu, torch.sqrt(torch.exp(self.prior_W_logvar)))
        Wpost = Normal(self.W_mu, torch.sqrt(torch.exp(self.W_logvar)))
        KL = kl_divergence(Wpost, Wprior).sum()

        # KL = 0.5 * ( self.prior_W_logvar - (self.W_logvar) - 1 \
        #            + torch.exp((self.W_logvar) - self.prior_W_logvar) \
        #            + torch.exp(-self.prior_W_logvar) * torch.pow(self.W_mu - self.prior_W_mu, 2) ).sum()

        # print('KL = {}; KL2 = {}'.format(KL.item(), KL2.item()))

        FE = (1 / batch_size) * LL - (1 / (seq_length * batch_size)) * KL

        return FE, (1 / batch_size) * LL, (1 / (seq_length * batch_size)) * KL

    def mc_prediction(self, init_X, A):
        """
        init_X : [1, output]
        A : [seq-1, 1, action_dim]
        """
        init_X = init_X.to(self.device)
        A = A.to(self.device)
        total_list = []
        total_o_list = []

        for i in range(500):
            # W_eps = torch.FloatTensor(init_X.size(0), self.W_logvar.shape[0], self.W_logvar.shape[1]).normal_().to(self.device)

            pred, _, _, o_matrix = self.rollout(init_X, A, track_o=True)  # [seq-1, 1, output]
            # x_sample = self.reparametrise(output_mean.squeeze(), torch.sqrt(output_sigma).squeeze())
            # total_list.append(x_sample.unsqueeze(-1))           #[seq-1, output, 1]
            total_list.append(pred.permute(0, 2, 1))  # [seq-1, output, 1]
            total_o_list.append(o_matrix.mean(-1))

        total_list = torch.cat(total_list, dim=-1)  # [seq-1, output, 300]
        mean = torch.mean(total_list, dim=-1)  # [seq-1, output]
        std = torch.std(total_list, dim=-1)  # [seq-1, output]

        total_o_list = torch.cat(total_o_list, dim=-1)  # [seq-1, 500]
        mean_o = total_o_list.mean(-1)
        std_o = total_o_list.std(-1)

        return mean, std, mean_o, std_o

    def uncertainty(self, init_X, A, object):
        """
        init_X : [1, output]
        A : [seq-1, 1, action_dim]
        """
        init_X = init_X.to(self.device)
        A = A.to(self.device)

        total_list = []
        for i in range(500):
            if object == 'W':
                pred, _, _ = self.rollout(init_X, A, W_uncertainty=True, epsilon_uncertainty=False)
            elif object == 'e':
                pred, _, _ = self.rollout(init_X, A, W_uncertainty=False, epsilon_uncertainty=True)
            else:
                print('Either W or e')

            total_list.append(pred.permute(0, 2, 1))
        total_list = torch.cat(total_list, dim=-1)
        mean = torch.mean(total_list, dim=-1)
        std = torch.std(total_list, dim=-1)

        return mean, std


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
            print('Zero occurs after squaring the sigma matrix. (batched gaussian ll)')

        inv_diag_cov = self.diagonalise(1 / (sigma ** 2),
                                        batch=True)  # a 2d batched matrix----> 3d batched diagonal tensor

        exp = ((x - mean).unsqueeze(-2)) @ inv_diag_cov @ ((x - mean).unsqueeze(-1))  #
        exp = exp.squeeze()  # [batch]
        # print(exp)

        # if 0 in torch.prod(cov, dim = -1):
        #    print('Zero occurs when calculating determinant of diagonal covariance. (batched gaussian ll)')

        logdet = torch.sum(2 * torch.log(sigma), dim=-1)
        # logdet = torch.log(torch.prod(sigma**2, dim = -1))         #product of all diagonal variance for each batch, shape [batch]
        # print('logdet=', logdet)
        n = mean.size()[-1]

        return -(n / 2) * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * exp

    def kl_divergence(self, prior_m, prior_sigma, post_m, post_sigma):
        """
        KL( q || p )
        shape : [hidden, hidden+1]
        """
        if 0 in prior_sigma ** 2:
            print('Zero occurs in squaring prior sigma')
        if 0 in post_sigma ** 2:
            print('Zero occurs in squaring posterior sigma')

        multi_normal_prior = MultivariateNormal(vec(prior_m), self.diagonalise(prior_sigma ** 2, False))
        multi_normal_post = MultivariateNormal(vec(post_m), self.diagonalise(post_sigma ** 2, False))

        return KL_f(multi_normal_post, multi_normal_prior)



    def reparametrise(self, mean, sigma):
        """
        sigma should have the same shape as mean (no correaltion)
        """
        # eps = torch.FloatTensor(sigma).normal_().to('cpu')

        eps = torch.rand_like(sigma).normal_().to(self.device)
        return mean + sigma * eps

    def stack_W(self, batch_size, mean, sigma):
        list_of_W = []
        for i in range(batch_size):
            temp_W = self.reparametrise(mean, sigma).unsqueeze(0)  # [1, hid, hid+1]

            list_of_W.append(temp_W)
        return torch.cat(list_of_W)  # [batch_size, hid, hid+1]

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





