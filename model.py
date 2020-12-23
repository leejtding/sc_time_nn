import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.input_size = 47192
        self.num_classes = 4
        self.batch_size = 20

        self.learning_rate = 0.001
        self.criterion = nn.CrossEntropyLoss()
        self.loss_list = []

        self.lin1 = nn.Linear(self.input_size, 10000)
        self.lin2 = nn.Linear(10000, 1000)
        self.lin3 = nn.Linear(1000, 100)
        self.lin4 = nn.Linear(100, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, input):
        output = F.relu(self.lin1(input))
        output = F.relu(self.lin2(output))
        output = F.relu(self.lin3(output))
        return self.lin4(output)

    def loss(self, outputs, labels):
        return torch.mean(self.criterion(outputs, labels.long()))

    def accuracy(self, logits, labels):
        total = labels.size(0)
        correct_predictions = torch.eq(torch.argmax(logits, 1), labels).sum().item()
        return correct_predictions / total
