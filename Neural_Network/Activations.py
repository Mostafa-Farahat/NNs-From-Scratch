import numpy as np
class Sigmoid():

    def __init__(self):
        pass

    def activate(self, z):
        return 1.0/(1+np.exp(-z))
    
    def activate_prime(self, z):
        return self.activate(z)*(1-self.activate(z))

class TanH():
    def __init__(self):
        pass

    def activate(self,z):
        return np.tanh(z)
    
    def activate_prime(self,z):
        tanh = np.tanh(z)
        return 1 - (tanh*tanh)

