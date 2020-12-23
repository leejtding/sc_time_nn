import math
import time
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.onnx
import torch.optim as optim

from preprocess import get_data
from model import Linear

def train(model, train_cells, train_labels):
    num_examples = len(train_labels)
    shuffle_indices = torch.randperm(num_examples)
    shuffle_labels = train_labels[shuffle_indices]
    shuffle_cells = train_cells[shuffle_indices]

    total_loss = 0
    start_time = time.time()
    num_batches = num_examples // model.batch_size
    for si in range(num_batches):
        starting_index = si * model.batch_size
        batch_indices = list(range(starting_index, (starting_index + model.batch_size)))
        batch_cells = shuffle_cells[batch_indices]
        batch_labels = shuffle_labels[batch_indices]

        model.optimizer.zero_grad()
        outputs = model(batch_cells)
        loss = model.loss(outputs, batch_labels)
        loss.backward()
        model.optimizer.step()

        total_loss += loss
        if si % 5 == 0:
            print("Average Loss for Batch {}: {}".format(si, loss))
        model.loss_list.append(loss)

    end_time = time.time()
    print("Train Time: {} seconds".format(end_time - start_time))


def test(model, test_cells, test_labels):
    num_examples = len(test_labels)
    shuffle_indices = torch.randperm(num_examples)
    shuffle_labels = test_labels[shuffle_indices]
    shuffle_cells = test_cells[shuffle_indices]
    num_batches = len(test_labels) // model.batch_size

    total_accuracy = 0

    with torch.no_grad():
        for bi in range(num_batches):
            starting_index = bi * model.batch_size
            batch_indices = list(range(starting_index, (starting_index + model.batch_size)))
            batch_cells = shuffle_cells[batch_indices]
            batch_labels = shuffle_labels[batch_indices]
            logits = model(batch_cells)
            total_accuracy += model.accuracy(logits, batch_labels)

    return total_accuracy / num_batches


def visualize_loss(losses):
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.savefig('scrnn.png')
    # plt.show()


torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_cells, train_labels, test_cells, test_labels = get_data('data/hsmm_data.csv', 'data/hsmm_times.csv')

model = Linear()
epochs = 15
for i in range(epochs):
    print("Epoch {}".format(i))
    train(model, train_cells, train_labels)
    print("\n")
visualize_loss(model.loss_list)

accuracy = test(model, test_cells, test_labels)
print("Model Accuracy: {}".format(accuracy))

