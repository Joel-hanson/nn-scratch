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

# Data = [[0, 0, 1],
#         [0, 0, 1],

#         [0, 1, 0],
#         [0, 1, 1],
#         [0, 1, 1],

#         [1, 0, 0],
#         [1, 0, 1],

#         [1, 1, 0],
#         [1, 1, 0],

#         [0, 0, 0],
#         [1, 1, 1]]
# Result = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]


Data = [[0, 0, 1],

        [0, 1, 0],
        [0, 1, 1],

        [1, 0, 0],
        [1, 0, 1],

        [1, 1, 0],

        [0, 0, 0],
        [1, 1, 1]]
Result = [1, 0, 1, 0, 1, 0, 0, 1]



# class NN1(object):
#     '''with 2 layers'''
#     def __init__(self, n_layers, x, y):
#         self.n_layers = n_layers
#         self.x = x
#         self.y = y
#         self.weights = [[0.0000001],[0.0000001],[0.0000001]]
#         self.bias = 0
#         self.output = 0
#         self._y =0


#     def gradient(self, epochs=10,learning_rate=0.001,*args):
#         N = float(len(self.y))
#         for _ in range(epochs):
#             for i,value in enumerate(self.x):
#                 self._y = np.dot( value,self.weights) + self.bias
#                 cost = sum([data**2 for data in (self.y[i]-self._y)]) / N
#                 w_gradient = -(2/N) * sum(value * (self.y[i] - self._y))
#                 b_gradient = -(2/N) * sum(self.y[i] - self._y)
#                 self.weights = self.weights - (learning_rate * w_gradient)
#                 self.bias = self.bias - (learning_rate * b_gradient)
#         return self.weights, self.bias, cost

#     def predict(self, test):
#         predicted_data = np.dot(test,self.weights) + self.bias
#         return predicted_data


class NN2(object):
    def __init__(self, n_layers, x, y):
        self.n_layers = n_layers
        self.x = x
        self.y = y
        self.weights = [[0.0000001],[0.0000001],[0.0000001]]
        self.bias = 0
        self.output = 0
        self._y =0


    def gradient(self, epochs=10,learning_rate=0.001,*args):
        for _ in range(epochs):
            for i,value in enumerate(self.x):
                self._y =  1 / (1 + np.exp(-np.dot(value,self.weights))) 
                error = self.y[i] - self._y
                self.weights += np.dot(value[np.newaxis].T,error*(self._y * (1 - self._y)[np.newaxis]))
        return self.weights

    def predict(self, test):
        predicted_data = np.dot(test,self.weights)
        return predicted_data


nn = NN2(1,np.array(Data),np.array(Result))
print(nn.gradient(100))
asd = [int(i) for i in input().split(',')]
print(nn.predict(np.array(asd)))
