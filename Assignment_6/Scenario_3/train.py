import torch
from torchvision import datasets
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from model import Net
from utils import get_transforms, train, test
import matplotlib.pyplot as plt
from torchsummary import summary
from utils import set_seed

def main():
    # Set seed for reproducibility
    set_seed(42)  # You can change this seed value
    
    # CUDA setup
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    # Get transforms
    train_transforms, test_transforms = get_transforms()
    
    # Load datasets
    train_data = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    
    # Dataloader arguments
    dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_data, **dataloader_args)
    
    # Model
    model = Net().to(device)

    print(summary(model, (1, 28, 28)))
    
    # Training setup
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    EPOCHS = 15
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.02,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=25,
        final_div_factor=1e4,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95
    )
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        epoch_train_losses, epoch_train_acc = train(model, device, train_loader, optimizer, scheduler, epoch)
        epoch_test_loss, epoch_test_acc = test(model, device, test_loader)
        
        train_losses.extend(epoch_train_losses)
        train_accuracies.extend(epoch_train_acc)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)
    
    # Plot results
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

def plot_metrics(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()

if __name__ == '__main__':
    main() 