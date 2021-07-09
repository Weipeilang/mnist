#encoding=utf-8
from bp import neuralNetwork
import numpy as np
import random
from PIL import Image
import mnist2csv
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

Net = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_images,train_labels = mnist2csv.load_mnist(r"../../MNIST/pytorch/data/MNIST/raw","train")
test_images,test_labels = mnist2csv.load_mnist(r"../../MNIST/pytorch/data/MNIST/raw","t10k")

epochs = 3

for e in range(epochs):
    for i in range(len(train_images)):
        inputs = (np.array(train_images[i]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes)
        targets[int(train_labels[i])] = 1
        Net.train(inputs, targets)
        if i % 1000 == 0:
            scorecard = []
            for j in range(100):
                correct_label = int(test_labels[j])
                inputs = (np.asfarray(test_images[j]) / 255.0 * 0.99) + 0.01
                outputs = Net.test(inputs)
                label = np.argmax(outputs)
                if (label == correct_label):
                    scorecard.append(1)
                else:
                    scorecard.append(0)
            scorecard_array = np.asarray(scorecard)
            print("epoch: {}   record:{}    正确率 ={}".format(e,i,scorecard_array.sum() / scorecard_array.size))


scorecard = []
for j in range(len(test_images)):
    correct_label = int(test_labels[j])
    inputs = (np.asfarray(test_images[j]) / 255.0 * 0.99) + 0.01
    outputs = Net.test(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
scorecard_array = np.asarray(scorecard)
print("在所有的测试集上面的正确率为 {:.4f}".format(scorecard_array.sum() / scorecard_array.size))