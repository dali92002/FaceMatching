#!/usr/bin/env python3
from ast import arg
from pathlib import Path
from data import TripletFaceDataset
import torch
import torch.optim as optim
from models.FaceMatch import FaceEmb
import os

# Implement functions here
def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    A method to train the model for one epoch

    Parameters:
            model: your created model
            device: specify the GPU or CPU
            train_loader: dataloader for training set
            optimizer: the traiining optimizer
            criterion: the loss function
            epoch: current epoch (int)
    """
    log_interval = 20 # specify to show logs every how many iterations 
    model.train()
    running_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        out_anchor, out_positive, out_negative = model(anchor, positive, negative) # get the embeddings
        loss = criterion(out_anchor, out_positive, out_negative) # compute the loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # print the logs
        if batch_idx % log_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), running_loss/log_interval))
            running_loss = 0

def validate(model, device, test_loader, criterion):
    """
    a method to validate the model

    Parameters:
            model: your created model
            device: specify to use GPU or CPU
            test_loader: The dataloader for testing
            criterion: the loss function
    
    """
    model.eval()
    test_loss = 0
    iters = 0
    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            out_anchor, out_positive, out_negative = model(anchor, positive, negative)
            test_loss += criterion(out_anchor, out_positive, out_negative)
            iters += 1
        test_loss /= iters
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        if not os.path.exists('./weights/'):
            os.makedirs('./weights/')
        torch.save(model.state_dict(), "./weights/model.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train_set', type=Path)
    parser.add_argument('--path_test_set', type=Path)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()
    
    # Implement execution here
    batch_size = args.batch_size 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build the training data loader
    train_dataset = TripletFaceDataset(args.path_train_set)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    # Build the testing Data loader
    test_dataset = TripletFaceDataset(args.path_train_set)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    # define your model
    model = FaceEmb()
    model = model.to(device)
    # define the loss and optimizer
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train and validate after each epoch (or specify it to be after a desired number)
    for epoch in range(1, 30+1):
        train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)
        validate(model, device, testloader, criterion)