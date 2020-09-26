import torch
import numpy as np
import math
import torch.nn as nn



class DRNN(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim, device, mode):
        super(DRNN, self).__init__()

        self.mode = mode

        self.init_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        if mode == 'RNN':
            self.recurrent = nn.RNN(action_dim, hidden_dim)
        elif mode == 'LSTM':
            self.recurrent = nn.LSTM(action_dim, hidden_dim)
        elif mode == 'GRU':
            self.recurrent = nn.GRU(action_dim, hidden_dim)
        else:
            raise ValueError('Mode must be one of RNN, LSTM and GRU')

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.device = device

    def forward(self, init_x, actions):
        """
        init_x : [batch, output_dim]
        actions : [seq, batch, input_dim]
        """

        init_x = init_x.to(self.device)
        actions = actions.to(self.device)

        init_h = self.init_encoder(init_x).unsqueeze(0).to(self.device)  # [1, batch, hidden]

        # print('action dim=', actions.shape, 'init_h dim=',init_h.shape)
        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)  # zero inital cell state
            recurrent_states = self.recurrent(actions, (init_h, init_c))[0]
        else:
            recurrent_states = self.recurrent(actions, init_h)[0]  # list of rnn hidden state

        output_list = []
        for h in recurrent_states:
            temp_out = self.decoder(h.squeeze(0))  # [batch, output_dim]

            # print('temp_out dim', temp_out.shape)
            output_list.append(temp_out.unsqueeze(0))

        return output_list


    def imagine(self, init_x, control_f, horizon, plan):
        """
        Given an initial state and the policy function, do model rollout and output sequence of trajectory
        init_x : [batch, output]
        """
        init_x = init_x.unsqueeze(0).to(self.device)  # [1,batch,output]

        init_h = self.init_encoder(init_x).to(self.device)  # [1, batch, hid]
        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)
            previous_c = init_c

        previous_x = init_x.squeeze(0)  # [batch, output]
        previous_h = init_h  # [1, batch, hid]
        output_list = []
        action_log_prob_list = []
        action_list = []
        for t in range(horizon):
            if plan == 'pg':
                action_dist = control_f(previous_x)  # [batch, 1]
                action_samples = action_dist.sample()  # [batch, 1]

                # compute log prob
                action_log_prob = action_dist.log_prob(action_samples)  # [batch, 1]
                action_log_prob_list.append(action_log_prob.unsqueeze(0))  # [1, batch, 1]
            elif plan == 'rp':
                action_samples, _ = control_f(previous_x)  # [batch, 1]
                action_log_prob_list = 0
                action_list.append(action_samples)

            if self.mode == 'LSTM':
                next_h, next_c = self.recurrent(action_samples.unsqueeze(0), (previous_h, previous_c))[1]
            else:
                next_h = self.recurrent(action_samples.unsqueeze(0), previous_h)[1]

            next_x = self.decoder(next_h.squeeze(0))  # [batch, output_dim]
            output_list.append(next_x.unsqueeze(0))  # [1, batch, output_dim]

            previous_h = next_h
            previous_x = next_x  # [batch,. output]
            if self.mode == 'LSTM':
                previous_c = next_c

        output_list = torch.cat(output_list)  # [seq-1, batch, output_dim ]
        if plan == 'pg':
            action_log_prob_list = torch.cat(action_log_prob_list)  # [seq-1, batch, 1]

        return output_list, action_list  # action_log_prob_list



    def validate_by_imagination(self, init_x, control_f, plan):

        init_x = init_x.unsqueeze(0).to(self.device)  # [1,batch,output]

        init_h = self.init_encoder(init_x).to(self.device)  # [1, batch, hid]
        if self.mode == 'LSTM':
            init_c = torch.zeros_like(init_h).to(self.device)
            previous_c = init_c

        previous_x = init_x.squeeze(0)  # [batch, output]
        previous_h = init_h  # [1, batch, hid]

        reward = 0
        iter = 0
        while True:
            action_samples, _ = control_f(previous_x)  # [batch, 1]
            if self.mode == 'LSTM':
                next_h, previous_c = self.recurrent(action_samples.unsqueeze(0), (previous_h, previous_c))[1]
            else:
                next_h = self.recurrent(action_samples.unsqueeze(0), previous_h)[1]

            next_x = self.decoder(next_h.squeeze(0))  # [batch, output_dim]

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
            previous_h = next_h
            previous_x = next_x

        return reward







