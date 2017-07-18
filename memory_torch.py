

import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, memory_size, key_size, choose_k = 256, inverse_temp = 40, margin = 0.1):
        super(Memory, self).__init__()
        self.keys = nn.Parameter(torch.Tensor(memory_size, key_size))
        nn.init.uniform(self.keys, a=-0.0, b=0.0)
        self.value = nn.Parameter(torch.Tensor(memory_size).zero_())
        self.age = nn.Parameter(torch.Tensor(memory_size).zero_())
        self.choose_k = choose_k
        self.inverse_temp = inverse_temp
        self.margin = margin
        self.bias = None
        self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.keys.data.uniform_(-0.1, 0.1)


    def forward(self, input, calc_cosine = False):
        """
        Computes nearest neighbors of the input queries.

        Arguments:
            input: A normalized matrix of queries of size num_inputs x key-size.
        Returns:

        """

        softmax_vals = None

        #Find the k-nearest neighbors of the query
        key_scores = torch.mm(input, torch.t(self.keys.data))
        values, indices = torch.topk(key_scores, self.choose_k, dim = 1)

        if calc_cosine:

            row_i = input.size()[0]
            column_i = self.choose_k
            cosine_sims = torch.Tensor(row_i, column_i)

            #Calculate similarity values
            for i in range(row_i):
                for j in range(column_i):
                    cosine_sims[i][j] = torch.dot(input[i], self.keys.data[indices[i][j]])
            sims_t = nn.Parameter(self.inverse_temp * cosine_sims)
            softmax = nn.Softmax()
            softmax_vals = softmax(sims_t)

        # Determine V[n_1]
        main_value = self.value[indices[0]]

        return main_value



test_input = (torch.FloatTensor([[5, 3, 2], [4, 9, 7]]))
test_model = Memory(1000, 3)
test_model.forward(test_input, calc_cosine=True)