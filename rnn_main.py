import math
import time
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.onnx
import torch.optim as optim

from preprocess2 import get_data
from model import RNN

def train(model, train_input, train_output):
    num_examples = len(train_input)
    shuffle_indices = torch.randperm(num_examples)
    shuffle_input = torch.from_numpy(train_input[shuffle_indices])
    shuffle_output = torch.from_numpy(train_output[shuffle_indices])

    total_loss = 0
    start_time = time.time()
    for i in range(num_examples):
        input_i = shuffle_input[i]
        output_i = shuffle_output[i]

        model.optimizer.zero_grad()
        outputs, _ = model(input_i)
        loss = model.loss(outputs, output_i)
        loss.backward()
        model.optimizer.step()

        total_loss += loss
        if i % 5 == 0:
            print("Average Loss for Batch {}: {}".format(i, loss))
        model.loss_list.append(loss)

    end_time = time.time()
    print("Train Time: {} seconds".format(end_time - start_time))


def test(model, test_input, test_output):
    num_examples = len(test_input)
    shuffle_indices = torch.randperm(num_examples)
    shuffle_labels = torch.from_numpy(test_input[shuffle_indices])
    shuffle_cells = torch.from_numpy(test_output[shuffle_indices])

    total_mse = 0

    with torch.no_grad():
        for i in range(num_examples):
            input_i = shuffle_cells[i]
            output_i = shuffle_labels[i]
            logits, _ = model(input_i)
            total_mse += model.loss(logits, output_i)

    avg_mse = total_mse / num_examples
    # perplex = math.exp(avg_loss)

    return avg_mse


def visualize_loss(losses):
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('sc_rnn.png')
    # plt.show()


torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_input, train_output, test_input, test_output = get_data('data/hsmm_data.csv', 'data/hsmm_times.csv')

model = RNN()
epochs = 10
for i in range(epochs):
    print("Epoch {}".format(i))
    train(model, train_input, train_output)
    print("\n")
visualize_loss(model.loss_list)

mse = test(model, test_input, test_output)
print("Model Mean Square Error: {}".format(mse))

