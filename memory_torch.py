

import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, memory_size, key_size, choose_k = 256, inverse_temp = 40, margin = 0.1):
        super(Memory, self).__init__()
        self.keys = nn.Parameter(torch.Tensor(memory_size, key_size))
        self.value = nn.Parameter(torch.Tensor(memory_size))
        self.age = nn.Parameter(torch.Tensor(memory_size))
        self.choose_k = choose_k
        self.inverse_temp = inverse_temp
        self.margin = margin
        self.bias = None
        self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.keys.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        return Memory()(input, self.keys, self.bias)

x = Memory(3, 6)
print(x)