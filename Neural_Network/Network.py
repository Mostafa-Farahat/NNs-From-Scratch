from DataLoader import DataLoader
import numpy as np

class Network():
    def __init__(self, layers: list, lr, epochs, with_bias, activation_func):
        loader = DataLoader()
        self.train_data, self.test_data = loader.loadData()
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.with_bias = with_bias
        self.activation = activation_func

        self.biases = [np.random.randn(y,1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.layers[:-1], self.layers[1:])]
        self.batch_size = 10
    
    def feed_forward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = self.activation.activate(np.dot(weight, a) + bias)
        return a

    def gradient_decent(self):

        for ep in range(self.epochs):
            batches = [self.train_data[k:k+self.batch_size] for k in range(0,len(self.train_data), self.batch_size)]
            for batch in batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

                for x, y in batch:
                    delta_nabla_b , delta_nabla_w = self.backprop(x,y)
                    nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [w-(self.lr/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
                self.biases = [b-(self.lr/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x.reshape(-1,1)
        activations = [activation]
        zs =[]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation.activate(z)
            activations.append(activation)

        y_onehot = np.zeros((3, 1))
        y_onehot[int(y)] = 1

        delta = (activations[-1]-y_onehot) * self.activation.activate_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range (2, len(self.layers)):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(), delta) * self.activation.activate_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def train(self):
        self.gradient_decent()




            
