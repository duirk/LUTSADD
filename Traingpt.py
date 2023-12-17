import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
device = torch.device('cuda:0')
# Define the transformation for the training set
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

# Define the dataset
dataset = datasets.ImageFolder(root="./imagenes", transform=transform)

# Define the training and validation splits
train_data, test_data = torch.utils.data.random_split(
    dataset, [0.8, 0.2])

# Create a data loader for the training set
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=128, shuffle=True)

# Define the transformation for the validation set
validation_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

# Define the validation dataset
validation_dataset = datasets.ImageFolder(
    root="./imagenes/validation",
    transform=validation_transform
)

# Create a data loader for the validation set
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=128, shuffle=False)

# Define the model
model = models.resnet18(pretrained=True)

# Define the optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(50):
    # Train the model
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate the model on the validation set
        if i % 50 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in validation_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            print(f"Epoch {epoch + 1}: {accuracy}")

            # Save the checkpoint
            with open("./imagenes/checkpoint_epoch_{epoch}.pt", "wb") as f:
                torch.save(model.state_dict(), f)

# Allow the user to input a word
word = input("Enter a word: ")

# Save the word to a file
with open("word.txt", "w") as f:
    f.write(word)

# Modify the code to only allow PNG files
def is_png(file):
    return file.endswith(".png")

# Define the transformation for the training set
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(is_png),
])

# Define the dataset
dataset = datasets.ImageFolder(root="./imagenes", transform=transform)
