import sys

sys.path.append("TrainModel/CNN/")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import MNISTFolderDataset
from model import MNIST_CNN
from trainer import train_model
from visualizer import plot_training_results
from evaluator import evaluate_model


def main():
    # Thiết lập thông số
    TRAIN_PATH = "TrainModel/Dataset/training"
    VAL_PATH = "TrainModel/Dataset/validation"
    TEST_PATH = "TrainModel/Dataset/testing"
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001

    # Xác định device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Định nghĩa transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load datasets
    train_dataset = MNISTFolderDataset(TRAIN_PATH, transform=transform)
    val_dataset = MNISTFolderDataset(VAL_PATH, transform=transform)
    test_dataset = MNISTFolderDataset(TEST_PATH, transform=transform)

    # Tạo dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo model
    model = MNIST_CNN().to(device)

    # Định nghĩa loss function và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device
    )

    # Vẽ kết quả training
    plot_training_results(train_losses, val_losses, train_accs, val_accs)

    # Load model tốt nhất và đánh giá
    model.load_state_dict(torch.load("TrainModel/CNN/ModelCNN/modelDL.pth"))
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
