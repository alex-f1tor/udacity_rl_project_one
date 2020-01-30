import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import torch.optim as optim
import torch.autograd as autograd 
USE_CUDA = torch.cuda.is_available()
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

hidden_size = 80
cuda0 = torch.device('cuda:0')
# class QNetwork(nn.Module):

#     def __init__(self, input_dim, output_dim, seed):
# #         """Initialize parameters and build model.
# #         Params
# #         ======
# #             state_size (int): Dimension of each state
# #             action_size (int): Dimension of each action
# #             s
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         self.feauture_layer = nn.Sequential(
#             nn.Linear(self.input_dim, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU()
#         )

#         self.value_stream = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )

#         self.advantage_stream = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, self.output_dim)
#         )

#     def forward(self, state):
#         features = self.feauture_layer(state)
#         values = self.value_stream(features)
#         advantages = self.advantage_stream(features)
#         qvals = values + (advantages - advantages.mean())
        
#         return qvals


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, seed):
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 =  nn.Linear(self.input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.noisy1 = NoisyLayer(hidden_size, self.output_dim)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.noisy1(x)
        return x
    
    def act(self, state):
        state   = torch.tensor(state, device=cuda0).unsqueeze(0)
        q_value = self.forward(state)
        action  = q_value.max(1)[1].data[0]
        return action
    
    
class NoisyLayer(nn.Module):
    def __init__(self, input_size, output_size, std = 0.017):
        super(NoisyLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.std = std
        mu_range = math.sqrt(3/hidden_size)

        self.mu_w = nn.Parameter(torch.tensor(torch.ones([self.input_size, self.output_size], device=cuda0).uniform_(-mu_range, mu_range), requires_grad=True))
        #self.mu_w.uniform_(-mu_range, mu_range)
        self.sigma_w = torch.tensor(torch.ones([self.input_size, self.output_size], device=cuda0)).fill_(self.std)
        self.epsilon_w = torch.tensor(torch.ones([self.input_size, self.output_size], device=cuda0)).uniform_(-1, 1)

        self.mu_b = nn.Parameter(torch.tensor(torch.ones([self.output_size], device=cuda0).uniform_(-mu_range, mu_range), requires_grad=True))
        #self.mu_b.
        self.sigma_b = torch.tensor(torch.ones([self.output_size], device=cuda0)).fill_(self.std)
        self.epsilon_b = torch.tensor(torch.ones([self.output_size], device=cuda0)).uniform_(-1, 1)
    def forward(self, x):
        if self.training:
            self.epsilon_w = torch.tensor(torch.ones([self.input_size, self.output_size], device=cuda0)).uniform_(-1, 1)
            self.epsilon_b = torch.tensor(torch.ones([self.output_size], device=cuda0)).uniform_(-1, 1)


            weight = self.mu_w + self.sigma_w.mul(self.epsilon_w)
            bias = self.mu_b + self.sigma_b.mul(self.epsilon_b)
        else: 
            weight = self.mu_w + self.sigma_w
            bias = self.mu_b + self.sigma_b

        return F.linear(x, torch.transpose(weight, 0, 1), bias)

     # def reset_parameters(self):
     #    mu_range = math.sqrt(3/self.weight_mu.size(1))
        
     #    self.mu_w.uniform_(-mu_range, mu_range)
     #    self.sigma_w.fill_(self.std_init )
        
     #    self.mu_b.uniform_(-mu_range, mu_range)
     #    self.sigma_b.fill_(self.std_init )

        


# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.017):
#         super(NoisyLinear, self).__init__()
        
#         self.in_features  = in_features
#         self.out_features = out_features
#         self.std_init     = std_init
        
#         self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
#         self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
#         self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
#         self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
#         self.reset_parameters()
#         self.reset_noise()
    
#     def forward(self, x):
#         if self.training: 
#             weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
#             bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
#         else:
#             weight = self.weight_mu
#             bias   = self.bias_mu
        
#         return F.linear(x, weight, bias)
    
#     def reset_parameters(self):
#         mu_range = math.sqrt(3/self.weight_mu.size(1))
        
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init )
        
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init )
    
#     def reset_noise(self):
#         epsilon_in  = self._scale_noise(self.in_features)
#         epsilon_out = self._scale_noise(self.out_features)
        
#         self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
#         self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
#     def _scale_noise(self, size):
#         x = torch.randn(size)
#         x = x.sign().mul(x.abs().sqrt())
#         return x