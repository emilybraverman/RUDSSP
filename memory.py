# Memory module as implemented in the paper:
#
# "Learning to Remember Rare Events",
# Published at ICLR 2017 by Google Brain.
#
# Authors: Lukasz Kaiser, Ofir Nachum, Aurko Roy, Samy Bengio
#
#
#



import numpy as np
import math
import torch
import torch.nn as nn

class Memory():
    """
    The memory module.
    """
    def __init__(self, key_size, memory_size, choose_k = 256, inverse_temp = 40, margin = 0.1):
        self.key_size = key_size
        self.memory_size = memory_size
        self.keys = np.random.randn(self.memory_size, self.key_size)
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
        #Find the k-nearest neighbors of the query
        values, indices = torch.topk(query.dot(self.keys.T), self.choose_k, dim = 0)

        #Calculate similarity values
        cosine_sims = [query.dot(self.keys[indices[i]]) for i in indices]
        sims_t = [x * self.inverse_temp for x in cosine_sims]
        softmax = nn.Softmax()
        softmax_vals = softmax(sims_t)

        # Determine V[n_1]
        main_value = self.value[indices[0]]

        return main_value, softmax_vals, indices

    def memory_loss(self, nearest_neighbors, query, ground_truth):
        """
        Calculates memory loss for a given query and ground truth label.

        Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the query in K.
            query: A normalized vector of size key-size.
            ground_truth: The correct desired label for the query.
        """

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

    def memory_update(self, main_value, query, ground_truth, indices):
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
            self.keys[indices[0]] = (q + self.keys[indices[0]]) / np.linalg.norm(q + self.k[indices[0]])
            self.age[indices[0]] = 0

            #Update age of everything else
            self.age = [self.age[x] + 1 for x in range(self.age) if x != 0]
        else:
            #Select n_prime, an index of maximum age that will be overwritten
            oldest = np.argwhere(self.age == np.amax(self.age))
            oldest = oldest.flatten().tolist()
            n_prime = np.random.choice(oldest)

            #Update at n_prime
            self.keys[n_prime] = query
            self.value[n_prime] = ground_truth
            self.age[n_prime] = 0

            #Update age of everything else
            self.age = [self.age[x] + 1 for x in range(self.age) if x != n_prime]

        return 0





