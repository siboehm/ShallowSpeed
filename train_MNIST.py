import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from mpi4py import MPI
from hashlib import sha1

torch.set_num_threads(1)
torch.manual_seed(0)

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


def get_model_hash(model):
    # this is probably not the most efficient way to do this, but it's
    # not straightforward to get a deterministic, content-based hash of a model
    hash_str = ""
    for param in model.parameters():
        numpy_param = param.data.cpu().numpy()
        # concat the strings to form a single hash later
        hash_str += sha1(numpy_param).hexdigest()
    # hash to concatenated strings
    return sha1(hash_str.encode("utf-8")).hexdigest()


def rprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)


def assert_sync(model_hash):
    # check that all processes have the same model hash
    model_hash_all = comm.gather(model_hash, root=0)
    if MPI.COMM_WORLD.rank == 0 and len(set(model_hash_all)) > 1:
        raise ValueError("Model hash mismatch")


if __name__ == "__main__":
    # init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("rank:", rank)

    # download the MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data/", train=True, download=True, transform=transform)
    # get a distinct subset of the dataset for each process
    dataset = Subset(
        dataset, torch.arange(start=rank, end=len(dataset), step=size, dtype=torch.long)
    )
    # Note: to make distributed training as similar as possible to serial training,
    # we need to turn of shuffling and make sure that `size` evenly divides the batch size
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=128 // size,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST("data/", train=False, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    # define the model
    model = MLP(input_size=28 * 28, hidden_size=64, output_size=10)
    # make sure the initialization is the same on all processes
    assert_sync(get_model_hash(model))

    # define the loss function
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train the model
    start_time = time.time()
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            if size > 1:
                for param in model.parameters():
                    recv = torch.zeros_like(param.grad)
                    comm.Allreduce(param.grad / size, recv, op=MPI.SUM)
                    param.grad = recv

            optimizer.step()
            if (i + 1) % 100 == 0:
                rprint(
                    "Epoch [{:2}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(train_loader),
                        loss.item(),
                    )
                )
        rprint("Time(epoch) {:4.1f}s".format(time.time() - epoch_start_time))
    rprint("Total training time: {:4.1f}s".format(time.time() - start_time))

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
        rprint(
            "Accuracy of the model on the 10000 test images: {} %".format(
                100 * correct / total
            )
        )

    # make sure the final model is the same on all processes
    assert_sync(get_model_hash(model))

    if rank == 0:
        torch.save(model.state_dict(), f"data/models/model_p{size}.pkl")
