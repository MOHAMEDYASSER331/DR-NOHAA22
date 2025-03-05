# DR-NOHAA22
import random
import math

#
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self):
      
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1)] 
        self.bias = random.uniform(-1, 1) 

    def forward(self, inputs):
     
        weighted_sum = inputs[0] * self.weights[0] + inputs[1] * self.weights[1] + self.bias

        output = sigmoid(weighted_sum)
        return output

    def train(self, inputs, target, learning_rate):
       
        output = self.forward(inputs)

        error = target - output

        gradient = error * sigmoid_derivative(output)

        self.weights[0] += inputs[0] * gradient * learning_rate
        self.weights[1] += inputs[1] * gradient * learning_rate
        self.bias += gradient * learning_rate

if __name__ == "__main__":
    training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]

    nn = NeuralNetwork()

    epochs = 10000  
    learning_rate = 0.1  

    print("Training the network...")
    for epoch in range(epochs):
        for inputs, target in training_data:
            nn.train(inputs, target, learning_rate)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}")
            for inputs, target in training_data:
                output = nn.forward(inputs)
                print(f"Input: {inputs}, Output: {output:.4f}, Target: {target}")

   
    print("\nTesting the trained network:")
    for inputs, target in training_data:
        output = nn.forward(inputs)
        print(f"Input: {inputs}, Output: {output:.4f}, Target: {target}")
