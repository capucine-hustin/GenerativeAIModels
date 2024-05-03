import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam


def modify_inception_v3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(weights=True, aux_logits=True)
    # Modify the final fully connected layer to output 10 categories
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        #print("Number of batches:", len(data_loader))    
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Preprocessing
    transform = transforms.Compose([
        transforms.Resize(299),   # resize to fit inception v3 input
        transforms.Grayscale(3),  # Convert single-channel image into three channels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load MNIST Dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Initializing model
    model = modify_inception_v3().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.002)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Save the model
    torch.save(model.state_dict(), 'mnist_inception_v3.pth')
