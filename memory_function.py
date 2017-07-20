import torch
import torch.autograd as ag
import torch.nn as nn
import numpy as np


class Memory(ag.Function):
    def __init__(self, memory_size, key_size, choose_k = 256, inverse_temp = 40, margin = 0.1, calc_cosine = False):
        self.memory_size = memory_size
        self.key_size = key_size
<<<<<<< HEAD

        #Initialize normalized key matrix
        keys = torch.randn((memory_size, key_size)).numpy()
        row_sums = keys.sum(axis=1)
        self.keys = nn.Parameter(torch.from_numpy(keys / row_sums[:, np.newaxis]))


        self.value = nn.Parameter(torch.from_numpy(np.array([i for i in range(memory_size)])))
        self.age = nn.Parameter(torch.from_numpy(np.zeros(memory_size)))
=======
        self.keys = nn.Parameter(torch.Tensor(memory_size, key_size))
        nn.init.uniform(self.keys, a=-0.0, b=0.0)
        self.value = nn.Parameter(torch.Tensor(memory_size).zero())
        self.age = nn.Parameter(torch.Tensor(memory_size).zero)()
>>>>>>> upstream/master
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
            input: A PyTorch Variable whose data contains queries of size batch_size x key-size.
        Returns:
            main_value, a batch-size x 1 matrix
        """

        #Normalize queries
        np_query = input.data.cpu().numpy()
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

<<<<<<< HEAD
def memory_loss(memory, ground_truth):
    """
    Calculates memory loss for a given memory and ground truth label.

    Arguments:
        memory: A Memory module.
        ground_truth: The correct desired labels for the queries.
=======
#    def memory_loss(self, query, ground_truth):
        """
#        Calculates memory loss for a given query and ground truth label.

        Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the queries in K.
            query: A normalized tensor of size batch-size x key-size.
            ground_truth: vector of size batch-size
        """#
        # batch-size x choose-k; elements are indices of key_scores dim 0
#        nearest_neighbors = self.nearest_neighbors
#        positive_neighbor = None
#        negative_neighbor = None

#        batch_indices = range(self.batch_indices[0], self.batch_indices[1])
        # batch_size x <256 matrix with all indices where query val != ground truth
#        negative_neighbor_indices = np.where(self.value[batch_indices, nearest_neighbors] != ground_truth)[0]
        #neg nbr = batch_size x 1 vector
#        negative_neighbor = negative_neighbor_indices[:, 0]

        #Flag notifying whether a positive neighbor was found
#        found = False

        #Find positive neighbor
        #for neighbor in nearest_neighbors:
        #    if self.value[neighbor] == ground_truth:
        #        positive_neighbor = neighbor
        #        found = True
        #        break
#        positive_neighbor_indices = np.where(self.value[batch_indices, nearest_neighbors] == ground_truth)[0]
        # pos nbr = batch_size x 1 vector#
#        positive_neighbor = positive_neighbor_indices[:, 0]

        #Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
#        if not found:
#            positive_neighbor = np.where(self.value == ground_truth)[0]

#        loss = query.dot(self.keys[negative_neighbor] - query.dot(self.keys[positive_neighbor]) + self.margin)

#        return loss
def memory_loss(memory, ground_truth):
    """
    Calculates memory loss for a given query and ground truth label.
    Arguments:
    		nearest_neighbors: A list of the indices for the k-nearest neighbors of the queries in K.
    		query: A normalized tensor of size batch-size x key-size.
    		ground_truth: vector of size batch-size
>>>>>>> upstream/master
    """
    nearest_neighbors = memory.nearest_neighbors
    queries = memory.queries
    positive_neighbor = None
    negative_neighbor = None
    batch_indices = range(memory.batch_indices[0], memory.batch_indices[1])
    loss = 0.0

    for query in range(queries.size()[0]):
        #Find negative neighbor
<<<<<<< HEAD
        for neighbor in nearest_neighbors[query]:
            if memory.value.data[neighbor] != ground_truth[query]:
                negative_neighbor = neighbor
                break

        #Flag notifying whether a positive neighbor was found
        if negative_neighbor == None:
            raise ValueError("Negative neighbor has none value: Your memory is full of the same value.")
        found = False

        #Find positive neighbor

        for neighbor in nearest_neighbors[query]:
            if memory.value.data[neighbor] == ground_truth[query]:
                positive_neighbor = neighbor
                found = True
                break

        #Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
        if not found:
            memory_vals = memory.value.data.numpy()
            positive_neighbor = np.where(memory_vals == ground_truth[query])[0][0]
        loss += torch.dot(queries[query], memory.keys.data[negative_neighbor]) - torch.dot(queries[query], memory.keys.data[positive_neighbor]) + memory.margin
=======
       for neighbor in nearest_neighbors[query]:
           if memory.value.data[neighbor] != ground_truth[query]:
               negative_neighbor = neighbor
               break

        #Flag notifying whether a positive neighbor was found
       found = False

        #Find positive neighbor

       for neighbor in nearest_neighbors[query]:
           if memory.value.data[neighbor] == ground_truth[query]:
               positive_neighbor = neighbor
               found = True
               break

        #Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
       if not found:
           memory_vals = memory.value.data.numpy()
           positive_neighbor = np.where(memory_vals == ground_truth[query])[0][0]

           loss += torch.dot(queries[query], memory.keys.data[negative_neighbor]) - torch.dot(queries[query], memory.keys.data[positive_neighbor]) + memory.margin

    return loss

def memory_loss_vectorized(memory, ground_truth):
    """
    Calculates memory loss for a given query and ground truth label.
    Arguments:
            nearest_neighbors: A list of the indices for the k-nearest neighbors of the queries in K.
            query: A normalized tensor of size batch-size x key-size.
            ground_truth: vector of size batch-size
    """
    nearest_neighbors = memory.nearest_neighbors
    queries = memory.queries
    positive_neighbor = None
    negative_neighbor = None
    batch_indices = range(memory.batch_indices[0], memory.batch_indices[1])
    loss = None

    # batch_size x <256 matrix with all indices where query val != ground truth
    negative_neighbor_indices = np.where(memory.value[batch_indices, nearest_neighbors] != ground_truth)[0]
    # neg nbr = batch_size x 1 vector
    negative_neighbor = negative_neighbor_indices[:, 0]

    # batch_size x <256 matrix with all indices where query val != ground truth
    positive_neighbor_indices = np.where(memory.value[batch_indices, nearest_neighbors] == ground_truth)[0]

    # a tensor of size batch size x choose k of booleans;
    bool_pos_nbr = np.equal(memory.value[batch_indices, nearest_neighbors], ground_truth.reshape(memory.batch_size, 1))
    # pos nbr = batch_size x 1 vector
    positive_neighbor = positive_neighbor_indices[:, 0]
    # Selects an arbitrary positive neighbor if none of the k-nearest neighbors are positive
    positive_neighbor[np.where(bool_pos_nbr==False)] = np.where(memory.value == ground_truth)[0]

    loss = queries.dot(memory.keys[negative_neighbor] - queries.dot(memory.keys[positive_neighbor]) + memory.margin)
>>>>>>> upstream/master

    return loss

def memory_update(memory, output, ground_truth):
    """
   Performs the memory update.
   Arguments:
       memory: A Memory Module.
        output: The resulting values from the forward pass through memory.
        ground_truth: The correct desired label for the query.

    """

    #unpack values
    queries = memory.queries
    indices = memory.nearest_neighbors

    for i in range(output.size()[0]):
        if output.data[i] == ground_truth[i]:
            n_1 = output.data[i]
            #Update key for n_1
            memory.keys.data[n_1] = (queries[i] + memory.keys.data[n_1]) / torch.norm(queries[i] + memory.keys.data[n_1])
            memory.age.data[n_1] = 0

            #Update age of everything else
            memory.age.data = torch.Tensor([memory.age.data[x] + 1 if x != n_1 else 0 for x in range(memory.age.data.size()[0])])
        else:
            #Select n_prime, an index of maximum age that will be overwritten
            oldest = np.where(memory.age.data.numpy() == np.max(memory.age.data.numpy()))[0]
            n_prime = np.random.choice(oldest)
            #Update at n_prime
            memory.keys.data[n_prime] = queries[i]
            memory.value.data[n_prime] = ground_truth[i]
            memory.age.data[n_prime] = 0

            #Update age of everything else
            memory.age.data = torch.from_numpy(np.asarray([memory.age.data[x] + 1 if x != n_prime else 0 for x in range(memory.age.data.size()[0])]))

    return 0

<<<<<<< HEAD


###### TEST ###########
# test_input = (torch.FloatTensor([[5, 3, 2], [4, 9, 7]]))
# test_truth = [10, 30]
# test_model = Memory(1000, 3)
# for i in range(450):
#     out = test_model.forward(test_input)
#     loss = memory_loss(test_model, test_truth)
#     print(i, ": ", loss)
#     memory_update(test_model, out, test_truth)
=======
test_input = (torch.FloatTensor([[5, 3, 2], [4, 9, 7]]))
test_truth = [10, 30]
test_model = Memory(1000, 3)
out = test_model.forward(test_input)
loss = memory_loss(test_model, test_truth)
print(loss)
memory_update(test_model, out, test_truth)
>>>>>>> upstream/master
