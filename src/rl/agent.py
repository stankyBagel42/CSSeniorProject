import copy
import random
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from src.rl.network import PokeNet


@dataclass
class AgentConfig:
    """Config for the agent, helps reduce arguments when constructing, and it's helpful for IDEs"""
    state_dim: int
    action_dim: int
    save_dir: Path = None
    batch_size: int = 32
    exploration_rate: float = 1.
    exploration_rate_decay: float = 0.99999975
    exploration_rate_min: float = 0.1
    gamma: float = 0.9
    warmup_steps: int = 1e3
    learn_freq: int = 3
    sync_freq: int = 1e4
    save_freq: int = 1e4
    lr: float = 0.00025
    base_nodes_layer: int = 64
    num_layers_per_side: int = 3
    memory_size: int = 100000
    use_argmax: bool = True


# made following https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#environment
class PokemonAgent:
    def __init__(self, agent_config: AgentConfig, checkpoint=None):

        self.cfg = agent_config

        # dimensions of arrays
        self.action_dim = self.cfg.action_dim
        self.memory = deque(maxlen=self.cfg.memory_size)
        self.batch_size = self.cfg.batch_size

        # model math settings
        self.exploration_rate = self.cfg.exploration_rate
        self.exploration_rate_decay = self.cfg.exploration_rate_decay
        self.exploration_rate_min = self.cfg.exploration_rate_min
        self.gamma = self.cfg.gamma

        # step frequencies
        self.curr_step = 0
        self.warmup_steps = self.cfg.warmup_steps
        self.learn_every = self.cfg.learn_freq
        self.sync_every = self.cfg.sync_freq
        self.save_every = self.cfg.save_freq

        self.save_dir = self.cfg.save_dir

        self.use_cuda = torch.cuda.is_available()

        # create networks and optimizer
        self.online_net = PokeNet(num_inputs=self.cfg.state_dim,
                                  num_outputs=self.cfg.action_dim,
                                  layers_per_side=self.cfg.num_layers_per_side,
                                  base_nodes=self.cfg.base_nodes_layer).float()
        self.target_net = copy.deepcopy(self.online_net)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg.lr)
        if self.use_cuda:
            self.online_net = self.online_net.to('cuda')
            self.target_net = self.target_net.to('cuda')
        if checkpoint:
            self.load(checkpoint)

        self.loss_fn = torch.nn.SmoothL1Loss()
        self.use_argmax = self.cfg.use_argmax

    def net(self, x, model):
        """Allows accessing both online and target networks at any time"""
        if model == 'online':
            return self.online_net(x)
        return self.target_net(x)

    def act(self, state) -> int:
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            if self.use_argmax:
                action_idx = torch.argmax(action_values, axis=1).item()
            else:
                action_idx = torch.distributions.Categorical(probs=action_values).sample().item()
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.warmup_steps:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.save_dir / f"PokeNet_{int(self.curr_step // self.save_every)}.pt"
        torch.save(
            dict(
                online_model=self.online_net.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                target_model=self.target_net.state_dict(),
                exploration_rate=self.exploration_rate,
                cfg=asdict(self.cfg)
            ),
            save_path
        )
        print(f"PokeNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path: str | Path):
        load_path = Path(load_path)
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        model_data = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = model_data['exploration_rate']
        online_state_dict = model_data['online_model']
        target_state_dict = model_data['target_model']
        optimizer_state_dict = model_data['optimizer_state']

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.online_net.load_state_dict(online_state_dict)
        self.target_net.load_state_dict(target_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.exploration_rate = exploration_rate
