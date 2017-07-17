



import torch
import torch.autograd as ag
import torch.nn as nn
import numpy as np


class Memory(ag.Function):
    def __init__(self, memory_size, key_size, choose_k = 256, inverse_temp = 40, margin = 0.1, calc_cosine = False):
        self.memory_size = memory_size
        self.key_size = key_size
        self.keys = nn.Parameter(torch.Tensor(memory_size, key_size))
        self.value = nn.Parameter(torch.Tensor(memory_size))
        self.age = nn.Parameter(torch.Tensor(memory_size))
        self.choose_k = choose_k
        self.inverse_temp = inverse_temp
        self.margin = margin
        self.calc_cosine = calc_cosine
        self.nearest_neighbors = torch.Tensor(memory_size, choose_k)
        self.queries = None

    def forward(self, input):
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
        self.nearest_neighbors = indices
        self.queries = input

        if self.calc_cosine:

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
        main_value = self.value[indices[:,0]]

        return main_value

    def backward(self, grad_output):
        pass

    def memory_loss(self, nearest_neighbors, query, ground_truth):
        """
        Calculates memory loss for a given query and ground truth label.

        Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the queries in K.
            query: A normalized vector of size key-size.
            ground_truth: The correct desired label for the query.
        """
        nearest_neighbors = self.nearest_neighbors
        positive_neighbor = None
        negative_neighbor = None
        #Find negative neighbor
        for neighbor in nearest_neighbors:
            if self.value[neighbor] != ground_truth:
                negative_neighbor = neighbor
                break

        #Flag notifying whether a positive neighbor was found
        found = False

        #Find positive neighbor
        for neighbor in nearest_neighbors:
            if self.value[neighbor] == ground_truth:
                positive_neighbor = neighbor
                found = True
                break

        #Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
        if not found:
            positive_neighbor = np.where(self.value == ground_truth)[0]

        loss = query.dot(self.keys[negative_neighbor] - query.dot(self.keys[positive_neighbor]) + self.margin)

        return loss

def memory_update(model, output, queries, ground_truth, indices):
    """
   Performs the memory update.

   Arguments:
       main_value: The value in the first index of self.value
        query: A normalized vector of size key-size.
        ground_truth: The correct desired label for the query.
        indices: A list of the k-nearest neighbors of the query.
    """
    if main_value == ground_truth:
        #Update key for n_1
        model.keys[indices[0]] = (q + model.keys[indices[0]]) / np.linalg.norm(q + model.keys[indices[0]])
        model.age[indices[0]] = 0

        #Update age of everything else
        model.age = [model.age[x] + 1 for x in range(model.age) if x != 0]
    else:
        #Select n_prime, an index of maximum age that will be overwritten
        oldest = np.argwhere(model.age == np.amax(model.age))
        oldest = oldest.flatten().tolist()
        n_prime = np.random.choice(oldest)

        #Update at n_prime
        model.keys[n_prime] = query
        model.value[n_prime] = ground_truth
        model.age[n_prime] = 0

        #Update age of everything else
        model.age = [model.age[x] + 1 for x in range(model.age) if x != n_prime]

    return 0

test_input = (torch.FloatTensor([[5, 3, 2], [4, 9, 7]]))
test_truth = [1, 3]
test_model = Memory(1000, 3)
out = test_model.forward(test_input)
memory_update()