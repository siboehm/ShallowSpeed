import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


# Define the MLP classifier
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # download the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/", train=False, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    # define the model
    model = MLP(input_size=28 * 28, hidden_size=64, output_size=10)
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train the model
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, 10, i + 1, len(train_loader), loss.item()
                    )
                )

    # test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(
            "Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )
    # save the model
    torch.save(model.state_dict(), "data/models/model.pkl")
