import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import random
import json

def load_data(path):
    # Unpickle data
    with gzip.open(path, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

        train_values = np.eye(np.max(training_data[1] + 1))[training_data[1]]
        valid_values = np.eye(np.max(validation_data[1] + 1))[validation_data[1]]
        test_values = np.eye(np.max(test_data[1] + 1))[test_data[1]]

        # Preproccess just a little
        train = [(x.reshape(training_data[0].shape[1], 1), train_values[i].reshape(10 ,1)) for i, x in enumerate(training_data[0])]
        valid = [(x.reshape(validation_data[0].shape[1], 1), valid_values[i].reshape(10, 1)) for i, x in enumerate(validation_data[0])]
        test = [(x.reshape(test_data[0].shape[1], 1), test_values[i].reshape(10, 1)) for i, x in enumerate(test_data[0])]

        return train, valid, test

class Network:
    def __init__(self, layers):
        self.size = len(layers)
        self.layers = layers
        # Randomize weights and biases: by given 'layers' input
        # corosponding to each layer apart from input

        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]

        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
    
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def fit(self, train, eta, lmbda, epochs, mini_batch_size, evaluation_data = None, no_learning_in = 10, min_eta = 128.0):
        cost = []
        max_match = 0
        no_improv = 0
        min_eta = eta / min_eta
        for q in range(epochs):
            random.shuffle(train)
            epoch_cost = []

            n = len(train)
            # Aquire mini batches as specified by mini_batch_size
            mini_batches = [train[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                # Initialize the delta weights and biases
                delta_weights_sum = [np.zeros(w.shape) for w in self.weights]
                delta_biases_sum = [np.zeros(b.shape) for b in self.biases]

                for m in range(len(mini_batch)):
                    # Definitions
                    batchX = mini_batch[m][0]
                    batchY = mini_batch[m][1]

                    # Feed the data forwards and remember each input to layer -> x
                    x = []
                    a = batchX
                    x.append(a)
                    for w, b in zip(self.weights, self.biases):
                        a = sigmoid(np.dot(w, a) + b)
                        x.append(a)

                    # Define the loss check
                    guess = a
                    y = batchY

                    # Create packpropegating errors
                    error = guess - y
                    delta_weights = []
                    delta_biases = []

                    loss = np.sum(np.nan_to_num(-y * np.nan_to_num(np.log(guess)) - (1 - y) * np.nan_to_num(np.log(1 - guess))))
                    epoch_cost.append(loss)

                    derv = error

                    delta_biases.append(derv)
                    delta_weights.append(np.dot(derv, x[-2].T))

                    # Calc the gradient and 'nudge' weights by it
                    for i in range(2, self.size):
                        derv = x[-i] * (1 - x[-i]) * np.dot(self.weights[-i + 1].T, derv)
                        delta_w = np.dot(derv, x[-i - 1].T)
                        delta_biases.append(derv)
                        delta_weights.append(delta_w)
                    delta_weights.reverse()
                    delta_biases.reverse()

                    # Sum over all the dw and db
                    delta_weights_sum = [nw+ow for nw, ow in zip(delta_weights, delta_weights_sum)]
                    delta_biases_sum = [nb+ob for nb, ob in zip(delta_biases, delta_biases_sum)]


                # 'Nudge' the weights and biases by the gradient descent
                # (dividing by the mini_batch_size to average the deltas)
                self.weights = [w * (1 - lmbda * eta / n) - (eta / mini_batch_size) * delta_weights_sum[j] for j, w in enumerate(self.weights)]

                self.biases = [b - (eta/mini_batch_size) * delta_biases_sum[j] for j, b in enumerate(self.biases)]
            
            # Remember the cost
            cost.append(np.average(epoch_cost))

            # Print the current stage of the network

            if evaluation_data:
                match, n = self.evaluate(evaluation_data)
                if match > max_match:
                    max_match = match
                    no_improv = 0
                else:
                    no_improv += 1
                    if no_improv == no_learning_in:
                        print("No improvment!")
                        eta /= 2.0
                        no_improv = 0
                        if eta <= min_eta:
                            return cost, q + 1
                print(f"Epoch #{q+1}: {match * 100 / n}% Winner: {max_match * 100 / n}%")
            else:
                print(f"Finished epoch #{q}")

        return cost, epochs

    def evaluate(self, test):
        n = len(test)
        match = 0
        for i in range(n):
            x = test[i][0]
            y = test[i][1]

            if np.argmax(self.feedforward(x)) == np.argmax(y):
                match += 1
        return match, n



def save_network(NN):
    data = {
        "layers": NN.layers,
        "weights": [w.tolist() for w in NN.weights],
        "biases": [b.tolist() for b in NN.biases]
    }
    with open("NeuralNetwork.json", "w") as net_file:
        json.dump(data, net_file)


def load_network(file):
    with open(file, "r") as net_file:
        data = json.load(net_file)
    net = Network(data["layers"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    return net
        
        
# Sigmoid activation func
def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))


if __name__ == "__main__":
    # load_data: raw_data -> train, values, test; assumes file path (string!) of pickled gz
    train, valid, test = load_data("mnist.pkl.gz")

    net = Network([784, 30, 10])

    end_cost, num_epochs = net.fit(train, eta=0.5, lmbda=5.0, epochs=10000, mini_batch_size=10, evaluation_data=valid)

    save_network(net)

    plt.figure()
    plt.plot(np.arange(0, num_epochs), end_cost)
    # plt.imshow(test[637][0].reshape(28,28), cmap='gray')
    plt.show()

    # # old_net = load_network("NeuralNetwork.pkl")
    # best_net = load_network("BestNetwork.json")

    # # match_o , n = old_net.evaluate(test)
    # match_b , n = best_net.evaluate(test)

    # # print(f"Old: {match_o} / {n}")
    # print(f"Best: {match_b} / {n} {match_b * 100 / n}%")

    # guess = old_net.feedforward(test[58][0])

    # print(f"Guess: {np.argmax(guess)}")
    # print(guess)
    # print(f"Number: {np.argmax(test[58][1])}")
