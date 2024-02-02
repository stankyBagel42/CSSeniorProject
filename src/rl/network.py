from torch import nn

class PokeNet(nn.Module):
    def __init__(self, num_inputs:int, num_outputs:int, layers_per_side:int=3):
        super().__init__()

        cur_count = 64
        layers = []

        for i in range(layers_per_side):
            inp = cur_count // 2 if i > 0 else num_inputs
            layers.append(nn.Linear(inp, cur_count, bias=True))
            layers.append(nn.ReLU(True))
            cur_count *= 2
        cur_count = cur_count // 2

        for i in range(layers_per_side):
            outp = num_outputs if i == layers_per_side-1 else cur_count // 2
            layers.append(nn.Linear(cur_count, outp, bias=True))
            layers.append(nn.ReLU(True))
            cur_count = cur_count // 2

        layers = layers[:-1]
        layers.append(nn.Softmax(dim=0))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


