import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import torch.optim as optim
from torchvision.ops import box_iou
from torch.utils.data import DataLoader

from datetime import datetime
import matplotlib.pyplot as plt

import pickle


def compute_accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            outputs = model(contexts)
            _, predicted = torch.max(outputs, dim=1)
            total += len(targets)
            correct += int((predicted == targets).sum())

    acc =  correct / total
    return acc

def plot_scores(model, model_performance, loss=True):
    if loss:
        label = 'Loss per epoch'
        y_label = 'Loss'
        title = f'Loss of {model.__class__.__name__} per epoch'
    else:
        label = 'Validation Accuracy'
        y_label = 'Accuracy'
        title = f'Accuracy of {model.__class__.__name__} per epoch'

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(len(model_performance)), model_performance, linestyle='-', label=label)


    # Vi set title and lables
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks([i for i, _ in enumerate(model_performance)][::5])
    ax.set_xticklabels([f'{i+1}' for i, _ in enumerate(model_performance)][::5], ha='right')
    ax.legend()

    plt.show()

def train(n_epochs, optimizer, model, train_loader, val_loader, loss_fn, device):

    n_batches_t = len(train_loader)

    # We'll store there the training loss for each epoch
    losses_train = []
    val_performance = []

    # We initialize the gradients to be zero before we begin to be sure they are zero
    optimizer.zero_grad(set_to_none=True)
    # We update weights for a number of epoks specified by the n_epoks parameter
    for epoch in range(1, n_epochs + 1):

        # Activating training mode (features such as dropout and batch normalization)
        # We have this inside the training loss because we calculate performance inside the loop
        model.train()

        loss_train = 0

        # We loop over the batches created by the train_loader
        for contexts, targets in train_loader:
            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            # We feed the batches to the model
            outputs = model(contexts)

            # Compute the accumulated loss
            loss = loss_fn(outputs, targets)

            # finds the gradients of the parameters
            loss.backward()

            # Takes one step of gradient decent
            optimizer.step()

            # zeroes out the gradients for the next round
            optimizer.zero_grad()

            # Adds the batch loss to the toalt loss of current epoch
            loss_train = loss_train + loss.item()

        # We finally store the loss of each epok
        losses_train.append(loss_train / n_batches_t)
        val_acc = compute_accuracy(model, val_loader, device=device)
        val_performance.append(val_acc)

        # We save the model after an epoch only if the validation set accuracy is
        # higher than the previous epoch

        if epoch == 1 or epoch % 1 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.3f}'.format(
                datetime.now().time(), epoch, loss_train / n_batches_t))

    return losses_train, val_performance
