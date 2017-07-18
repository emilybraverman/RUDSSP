import torch
import torch.autograd as ag
import torch.nn as nn
import numpy as np


class Memory(ag.Function):
    def __init__(self, memory_size, key_size, choose_k = 256, inverse_temp = 40, margin = 0.1, calc_cosine = False):
        self.memory_size = memory_size
        self.key_size = key_size

        #Initialize normalized key matrix
        keys = torch.Tensor(memory_size, key_size)
        self.keys = nn.Parameter(torch.Tensor(memory_size, key_size))


        self.value = nn.Parameter(torch.from_numpy(np.array([i for i in range(memory_size)])))
        self.age = nn.Parameter(torch.LongTensor(memory_size))
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
            input: A normalized matrix of queries of size choose_k x key-size.
        Returns:
            main_value, a batch-size x 1 matrix
        """

        #Normalize queries
        np_query = input.numpy()
        row_sums = np_query.sum(axis = 1)
        input = torch.from_numpy(np_query / row_sums[:, np.newaxis])


        softmax_vals = None

        #Find the k-nearest neighbors of the query
        key_scores = torch.mm(input, torch.t(self.keys.data))
        values, indices = torch.topk(key_scores, self.choose_k, dim = 1)
        self.nearest_neighbors = indices
        self.queries = input

        # Determine V[n_1]
        main_value = self.value[indices[:, 0]]

        if self.calc_cosine:
            # is this the condition for when the memory module is embedded?

            # Calculate similarity values
            cosine_sims = torch.dot(input, torch.t(self.keys[indices, :]))

            #row_i = input.size()[0]
            #column_i = self.choose_k
            #cosine_sims = torch.Tensor(row_i, column_i)

            #Calculate similarity values
            #for i in range(row_i):
            #    for j in range(column_i):
            #        cosine_sims[i][j] = torch.dot(input[i], self.keys.data[indices[i][j]])

            sims_t = nn.Parameter(self.inverse_temp * cosine_sims)
            softmax = nn.Softmax()
            softmax_vals = softmax(sims_t)

            return main_value, softmax_vals

        return main_value

    def backward(self, grad_output):
        pass

    def memory_loss(self, ground_truth):
        """
        Calculates memory loss for a given query and ground truth label.
        Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the queries in K.
            query: A normalized tensor of size batch-size x key-size.
            ground_truth: vector of size batch-size
        """

        #unpack values
        queries = self.queries

        # batch-size x choose-k; elements are indices of key_scores dim 0
        nearest_neighbors = self.nearest_neighbors
        positive_neighbor = None
        negative_neighbor = None
        #Find negative neighbor
        #for neighbor in nearest_neighbors:
        #    if self.value[neighbor] != ground_truth:
        #        negative_neighbor = neighbor
        #        break
        batch_indices = range(self.batch_indices[0], self.batch_indices[1])
        # batch_size x <256 matrix with all indices where query val != ground truth
        negative_neighbor_indices = np.where(self.value[batch_indices, nearest_neighbors] != ground_truth)[0]
        #neg nbr = batch_size x 1 vector
        negative_neighbor = negative_neighbor_indices[:, 0]

        #Flag notifying whether a positive neighbor was found
        found = False

        #Find positive neighbor
        #for neighbor in nearest_neighbors:
        #    if self.value[neighbor] == ground_truth:
        #        positive_neighbor = neighbor
        #        found = True
        #        break
        positive_neighbor_indices = np.where(self.value[batch_indices, nearest_neighbors] == ground_truth)[0]
        # pos nbr = batch_size x 1 vector
        positive_neighbor = positive_neighbor_indices[:, 0]

        #Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
        if not found:
            positive_neighbor = np.where(self.value == ground_truth)[0]

        loss = query.dot(self.keys[negative_neighbor] - query.dot(self.keys[positive_neighbor]) + self.margin)

        return loss

def memory_update(memory, output, ground_truth):
    """
   Performs the memory update.

   Arguments:
       main_value: The value in the first index of self.value
        query: A normalized vector of size key-size.
        ground_truth: The correct desired label for the query.
        indices: A list of the k-nearest neighbors of the query.
    """

    #unpack values
    queries = memory.queries
    indices = memory.nearest_neighbors

    for i in range(output.size()[0]):
        if output.data[i] == ground_truth[i]:
            n_1 = output[i]
            #Update key for n_1
            memory.keys.data[n_1] = (queries[i] + memory.keys.data[n_1]) / torch.norm(queries[i] + memory.keys.data[n_1])
            memory.age.data[n_1] = 0

            #Update age of everything else
            memory.age.data = torch.Tensor([memory.age.data[x] + 1 if x != n_1 else 0 for x in range(memory.age.data.size()[0])])
        else:
            #Select n_prime, an index of maximum age that will be overwritten
            max, n_prime_tensor = output.data.max(0)
            n_prime = n_prime_tensor[0]
            #Update at n_prime
            memory.keys.data[n_prime] = queries[i]
            memory.value.data[n_prime] = ground_truth[i]

            #Update age of everything else
            memory.age.data = torch.Tensor([memory.age.data[x] + 1 if x != n_prime else 0 for x in range(memory.age.data.size()[0])])

    return 0

test_input = (torch.FloatTensor([[5, 3, 2], [4, 9, 7]]))
test_truth = [10, 30]
test_model = Memory(1000, 3)
out = test_model.forward(test_input)
loss = test_model.memory_loss(test_truth)
print(loss)
memory_update(test_model, out, test_truth)
