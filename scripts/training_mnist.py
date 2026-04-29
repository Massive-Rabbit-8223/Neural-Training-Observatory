import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from neutrobs.utils.storage import DuckDBStore
from neutrobs.utils.observer import ObserverEngine, LossObserver, GradNormObserver, GradObserver, ActivationStatsObserver
from neutrobs.utils.tensor_processor import TensorProcessor
from neutrobs.utils.datatypes import TensorSummaryConfig
from neutrobs.utils.logger import WandBLogger

# ---------------------------
# Setup data
# ---------------------------
transform = transforms.Compose([transforms.ToTensor()])

training_data = torchvision.datasets.MNIST(
    root='../datasets', 
    train=True, 
    download=True, 
    transform=transform
)

test_data = torchvision.datasets.MNIST(
    root='../datasets', 
    train=False, 
    download=True, 
    transform=transform
)

print(len(training_data))
print(len(test_data))

# Hyper-parameters
batch_size = 64
learning_rate = 1e-3
epochs = 5

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


# ---------------------------
# Model Creation
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer, observer, epochs=1):
    step = 0
    size = len(dataloader.dataset)
    model.train()   # set model into training mode -> only affects certain modules
    for epoch in tqdm(range(epochs)):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            observer.emit("batch_start", step=step, epoch=epoch)    # NOT IMPLEMENTED !!!

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            observer.emit(
                "forward_end",
                step=step,
                loss=loss,
                get_model=lambda: model
            )

            # Backpropagation
            loss.backward()

            observer.emit(
                "backward_end",
                step=step,
                get_model=lambda: model
            )

            optimizer.step()
            optimizer.zero_grad()

            observer.emit("optimizer_step", step=step, epoch=epoch) # NOT IMPLEMENTED !!!
            
            step += 1

        observer.emit("epoch_end", epoch=epoch, step=step)          # NOT IMPLEMENTED !!!

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":

    config = {
        "lr": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "model": "MLP"
    }

    tensor_processor = TensorProcessor(
        TensorSummaryConfig(
            mean=True,
            std=True,
            p90=True,
            p99=True,
            sparsity=True
        )
    )

    store = DuckDBStore("observatory_test.duckdb")

    logger = WandBLogger(
        project="neural-training-observatory",
        run_name="mnist_run",
        config=config
    )

    observer = ObserverEngine(
        modules=[
            LossObserver(),
            GradNormObserver(100),
            GradObserver(100),
            ActivationStatsObserver(100)
        ],
        store=store,
        logger=logger,
        tensor_processor=tensor_processor,
        run_id="mnist_demo_run"
    )

    train(
        train_dataloader, 
        model, 
        loss_fn, 
        optimizer, 
        observer, 
        epochs
    )
    store.flush()
    observer.close()
    print("Done!")

    #torch.save(model.state_dict(), "model.pth")
    #print("Saved PyTorch Model State to model.pth")

    df = store.get_metric('loss')
