# Memory module as implemented in the paper:
#
# "Learning to Remember Rare Events",
# Published at ICLR 2017 by Google Brain.
#
# Authors: Lukasz Kaiser, Ofir Nachum, Aurko Roy, Samy Bengio


import numpy as np
import torch
import torch.nn as nn

class Memory():
    """
    The memory module.
    """
    def __init__(self, key_size, memory_size, choose_k = 256, inverse_temp = 40, margin = 0.1):
        self.key_size = key_size
        self.memory_size = memory_size
        self.keys = k
        self.value = np.zeros(memory_size)
        self.age = np.zeros(memory_size)
        self.choose_k = choose_k
        self.inverse_temp = inverse_temp
        self.margin = margin

    def query(self, query, ground_truth):
        """
        Computes nearest neighbors of the input queries.

        Arguments:
            query: A normalized vector of size key-size.
            ground_truth: The correct desired label for the query.
        Returns:

        """

        values, indices = torch.topk(query.repeat(self.memory_size).dot(self.keys.T), self.choose_k, dim = 0)
        cosine_sims = [query.dot(self.K[indices[i]]) for i in indices]
        sims_t = [x * self.inverse_temp for x in cosine_sims]
        softmax = nn.Softmax()
        softmax_vals = softmax(sims_t)

        # Determine V[n_1]
        main_value = self.V[indices[1]]

        return main_value, softmax_vals, indices

    def memory_loss(self, nearest_neighbors, query, ground_truth):
        """
        Calculates memory loss for a given query and ground truth label.

        Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the query in K.
            query: A normalized vector of size key-size.
            ground_truth: The correct desired label for the query.
        """
        for neighbor in nearest_neighbors:
            if self.value[neighbor] != ground_truth:
                negative_neighbor = neighbor
                break

        found = False
        for neighbor in nearest_neighbors:
            if self.value[neighbor] == ground_truth:
                positive_neighbor = neighbor
                found = True
                break
        if not found:
            positive_neighbor = np.where(self.value == ground_truth)[0]

        loss = query.dot(self.k[negative_neighbor] - query.dot(self.k[positive_neighbor]) + self.margin)

        return loss

    def memory_update(self, main_value, query, ground_truth, indices):
        if main_value == ground_truth:
            #Update key for n_1
            self.k[indices[0]] = (q + self.k[indices[0]]) / np.linalg.norm(q + self.k[indices[0]])
            self.age[indices[0]] = 0
        else:
            pass
