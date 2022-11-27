import torch
from dataset import create_dataset
import torch.optim as optim
import torchvision


if __name__ == "__main__":

    # define train_loader for dataset
    train_loader = create_dataset()

    
    # define model 
    net = torchvision.models.resnet50()

    # define optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=3, factor=0.3)


    # start training    
    for item in train_loader:
        a = 1