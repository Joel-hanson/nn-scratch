import numpy as np

'''
raining,friend,distance
Data = [[0,0,1],
        [0,0,1],

        [0,1,0],
        [0,1,1],
        [0,1,1],

        [1,0,0],
        [1,0,1],

        [1,1,0],
        [1,1,0],
        
        [0,0,0],
        [1,1,1]]
Result = [0,1, 1, 1,0, 0,0, 0,1, 1,0]
'''

Data = [[0, 0, 1],
        [0, 0, 1],

        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 1],

        [1, 0, 0],
        [1, 0, 1],

        [1, 1, 0],
        [1, 1, 0],

        [0, 0, 0],
        [1, 1, 1]]
Result = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]


# class NN(object):
#     def __init__(self,n_layers,x,y):
#         self.n_layers = n_layers
#         self.x = x
#         self.y = y
#         self.weights = {}
#         self.bias = {}
#         for i in range(n_layers):
#             self.weights[i] = np.ones((1,3))
#             self.bias[i] = np.zeros((1,3))
#         self.output     = np.zeros((1,3))
#         self._y = np.zeros((1,3))
    
#     def feedForward(self,epochs=10,):
#         for _ in self.n_layers:
#             self._y = (self.weights*self.x) + self.bias

#         N = float(len(self.y))
#         for _ in range(epochs):
#             y_current = (m_current * X) + b_current
#             cost = sum([data**2 for data in (y-y_current)]) / N
#             m_gradient = -(2/N) * sum(X * (y - y_current))
#             b_gradient = -(2/N) * sum(y - y_current)
#             m_current = m_current - (learning_rate * m_gradient)
#             b_current = b_current - (learning_rate * b_gradient)
#         return m_current, b_current, cost

#     def backProp(self):
#         for _ in self.n_layers:
#             loss = self._y -self.y


class NN1(object):
    '''with 2 layers'''
    def __init__(self, n_layers, x, y):
        self.n_layers = n_layers
        self.x = x
        self.y = y
        self.weights = 0.0000001
        self.bias = 0
        self.output = 0
        self._y =0


    def gradient(self, epochs=10,learning_rate=0.001,*args):
        N = float(len(self.y))
        import pdb; pdb.set_trace()
        for _ in range(epochs):
            for i,value in enumerate(self.x):
                self._y = (self.weights * value) + self.bias
                cost = sum([data**2 for data in (self.y[i]-self._y)]) / N
                w_gradient = -(2/N) * sum(value * (self.y[i] - self._y))
                b_gradient = -(2/N) * sum(self.y[i] - self._y)
                self.weights = self.weights - (learning_rate * w_gradient)
                self.bias = self.bias - (learning_rate * b_gradient)
        return self.weights, self.bias, cost

    def predict(self, test):
        predicted_data = (self.weights * test) + self.bias
        return predicted_data

nn = NN1(1,np.array(Data),np.array(Result))
nn.gradient(100)
print(nn.gradient())
