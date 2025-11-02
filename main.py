import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

import pathlib

# Create a directory for saving images
image_dir = pathlib.Path("images/movie_images")
image_dir.mkdir(parents=True, exist_ok=True)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.weights + self.bias



weight = 10
bias = 5
start = 0
end = 10
step = 0.01

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight*X + bias

# creating a train/test split

train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

def plt_predictions(train_data=X_train,
                    train_label=y_train,
                    test_data=X_test,
                    test_label=y_test,
                    save_path="predictions.png",
                    predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.savefig(save_path)


# 
loss_fn = nn.L1Loss()
torch.manual_seed(42)
model = LinearRegressionModel()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
# 

print(optimizer)

def train_model(model, epochs, graph_movie=False):
    epochs_count = []
    loss_values = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            y_test_pred = model(X_test)
            test_loss = loss_fn(y_test_pred, y_test)

        if epoch % 100 == 0:
            epochs_count.append(epoch)
            loss_values.append(loss.item())
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

        if graph_movie:
            epoch_formatted = f"{epoch:03d}"
            plt_predictions(predictions=model(X_test).detach(), save_path=image_dir / f"epoch_{epoch_formatted}.png")

    return epochs_count, loss_values, test_loss.item()

train_model(model, 950, graph_movie=True)
