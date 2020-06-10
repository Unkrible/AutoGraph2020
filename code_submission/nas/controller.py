import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNController(nn.Module):
    def __init__(self, num_tokens, hidden_dim):
        super(GNNController, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.n_actions = n_actions
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        # action index -> embedding
        self.encoder = nn.Embedding(sum(num_tokens), hidden_dim)

        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # embedding -> action logits
        self.decoders = nn.ModuleList()
        for num_token in num_tokens:
            decoder = nn.Linear(hidden_dim, num_token)
            self.decoders.append(decoder)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, states, action_i):
        hx, cx = self.lstm(inputs, states)
        logits = self.decoders[action_i](hx)

        return logits, (hx, cx)

    def sample(self, batch_size=1):
        # TODO: to device
        inputs, states = self.init_state(batch_size)
        entropies, log_probs, actions = [], [], []
        for block_idx, num_token in enumerate(self.num_tokens):
            logits, states = self(inputs, states, block_idx)
            probs = logits.softmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1)
            action = probs.multinomial(1).type(torch.long)
            # TODO: why use Variable
            # selected_log_prob = log_prob.gather(1, torch.autograd.Variable(torch.tensor(action), requires_grad=False))
            selected_log_prob = log_prob.gather(1, action)
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])
            # TODO: require_grad=False?
            inputs = (action[:, 0] + sum(self.num_tokens[:block_idx])).to(self.device)
            inputs = self.encoder(inputs)
            actions.append(action[:, 0])
        actions = torch.stack(actions).transpose(0, 1)
        return actions, torch.cat(log_probs), torch.cat(entropies)

    def init_state(self, batch_size):
        inputs = torch.zeros([batch_size, self.hidden_dim])
        states = (torch.zeros([batch_size, self.hidden_dim]), torch.zeros([batch_size, self.hidden_dim]))
        return inputs, states