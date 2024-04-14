import torch
from torch import nn

class PokeNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, layers_per_side: int = 3, base_nodes: int = 64):
        super().__init__()

        self.device = None
        cur_count = base_nodes
        layers = []

        for i in range(layers_per_side):
            inp = cur_count // 2 if i > 0 else num_inputs
            layers.append(nn.Linear(inp, cur_count, bias=True))
            layers.append(nn.ReLU(True))
            cur_count *= 2
        cur_count = cur_count // 2

        for i in range(layers_per_side):
            outp = num_outputs if i == layers_per_side - 1 else cur_count // 2
            layers.append(nn.Linear(cur_count, outp, bias=True))
            layers.append(nn.ReLU(True))
            cur_count = cur_count // 2

        layers = layers[:-1]

        self.softmax = nn.Softmax()

        switch_scalar = 0.5
        self.scale_switches = torch.Tensor([1, 1, 1, 1, switch_scalar, switch_scalar, switch_scalar, switch_scalar, switch_scalar])
        self.network = nn.Sequential(*layers)

    def to(self, device, **kwargs):
        self.device = device
        self.scale_switches = self.scale_switches.to(device)
        super().to(device, **kwargs)

    def cuda(self, device=None):

        self.scale_switches = self.scale_switches.cuda()
        return super().cuda(device)

    def cpu(self):
        self.scale_switches = self.scale_switches.cpu()
        return super().cpu()

    def forward(self, x):

        x = self.network(x)
        x = x * self.scale_switches
        return self.softmax(x)


class DuelingPokeNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, layers_per_side: int = 3, base_nodes: int = 64):
        super().__init__()
        self.device = None
        cur_count = base_nodes
        layers = []

        for i in range(layers_per_side):
            inp = cur_count // 2 if i > 0 else num_inputs
            layers.append(nn.Linear(inp, cur_count, bias=True))
            layers.append(nn.ReLU(True))
            cur_count *= 2
        cur_count = cur_count // 2

        for i in range(layers_per_side - 1):
            layers.append(nn.Linear(cur_count, cur_count // 2, bias=True))
            layers.append(nn.ReLU(True))
            cur_count = cur_count // 2

        # main network that embeds the state
        self.network = nn.Sequential(*layers)

        # layer to produce estimated state value from embedding
        self.state_val = nn.Linear(cur_count, 1)

        # layer to predict advantage from each action
        self.advantage = nn.Linear(cur_count, num_outputs)

        switch_scalar = 0.5
        self.scale_switches = torch.Tensor([1, 1, 1, 1, switch_scalar, switch_scalar, switch_scalar, switch_scalar, switch_scalar])
        self.softmax = nn.Softmax()

    def to(self, device, **kwargs):
        self.device = device
        self.scale_switches = self.scale_switches.to(device)
        return super().to(device, **kwargs)

    def cuda(self, device=None):

        self.scale_switches = self.scale_switches.cuda()
        return super().cuda(device)

    def cpu(self):
        self.scale_switches = self.scale_switches.cpu()
        return super().cpu()

    def forward(self, x):

        x = self.network(x)
        state_val = self.state_val(x)
        advantage = self.advantage(x)

        # subtract mean to make the function identifiable
        advantage = advantage + (state_val - advantage.mean())

        advantage = advantage * self.scale_switches

        # turn advantage into a probability distribution
        action_dist = self.softmax(advantage)

        return action_dist
