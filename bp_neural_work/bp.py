#encoding=utf-8
import numpy as np
import scipy.special

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        self.inodes = inputNodes #784
        self.hnodes = hiddenNodes #200
        self.onodes = outputNodes #10
        self.lr = learningrate

        self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5),(self.hnodes, self.inodes)) #200*784
        self.who = np.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes, self.hnodes)) #10*200

        # activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T  #784*1
        targets = np.array(targets_list, ndmin=2).T #10*1

        hidden_inputs = np.dot(self.wih, inputs)  #200*784  784*1
        hidden_outputs = self.activation_function(hidden_inputs)  #200*1

        final_inputs = np.dot(self.who, hidden_outputs)  #10*200 200*1
        final_outputs = self.activation_function(final_inputs) #10*1

        output_errors = targets - final_outputs
        g_i =  final_outputs * (1-final_outputs) * output_errors #10*1 10*1 10*1
        hidden_errors = np.dot(self.who.T, g_i)  # 200*10  10*1

        self.who += self.lr * np.dot(g_i,np.transpose(hidden_outputs))  #  1*200
        e_h = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        self.wih += self.lr * np.dot(e_h, np.transpose(inputs))

    def test(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


