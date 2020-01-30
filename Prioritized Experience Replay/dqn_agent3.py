import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
EPS = 0.00000001            # small const for non-zero transition-priority value
ALPHA = 0.7             # const for compute sample transition probability


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.priority = PriorityBuffer(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, beta=1):
        # Save experience in replay memory

        # Get max predicted Q values (for next states) from target model
        self.memory.add(state, action, reward, next_state, done)
 
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, importance = self.memory.sample(self.priority.priority, beta)
                self.learn(experiences, importance, GAMMA)

    def prioritize(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action = torch.from_numpy(np.array(action)).long().unsqueeze(0).to(device)
        reward = torch.from_numpy(np.array(reward)).float().unsqueeze(0).to(device)
        next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(device)
        done = torch.from_numpy(np.array(done, dtype=np.uint8)).float().unsqueeze(0).to(device)


        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_expected_per = self.qnetwork_local(state).detach().max(1)[0].unsqueeze(1)
         
            Q_target_next_per = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)
           
            Q_target_per = reward + (GAMMA* Q_target_next_per * (1 - done))
           
            delta = torch.abs(Q_target_per - Q_expected_per) + EPS


        self.priority.add((delta[0][0].cpu().numpy())**ALPHA+0.000000001)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selectionNavigation
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, importance, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_local_next_indexes = self.qnetwork_local(next_states).detach().max(1)[1]
       
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, Q_local_next_indexes.view(-1,1))
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = torch.mean(importance*(Q_expected-Q_targets)**2) #rmse by hands along axis
       
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

       
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PriorityBuffer:
    def __init__(self, buffer_size):
        self.priority = deque(maxlen=buffer_size)

    def add(self, delta_val):
        """Add a new experience to memory."""
        self.priority.append(delta_val)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, priority, beta):
        """Randomly sample a batch of experiences from memory."""

        #change delta to p, according prioritized experience replay
        
        p_list = np.array(list(priority))
     
        p_normed = np.array(p_list/np.sum(p_list))
        
        #select according weights p_normed
        sample_indexes = random.choices(range(len(p_normed)), k=self.batch_size, weights=p_normed)

        wi = np.array(1/(len(p_normed)*p_normed)**beta)
        wi_normed = np.array(wi/np.max(wi))[sample_indexes]

        importance = torch.from_numpy(wi_normed.reshape([-1,1])).float().to(device)
        experiences = [self.memory[i] for i in sample_indexes]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones), importance

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
