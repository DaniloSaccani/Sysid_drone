import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set input/output dimensions
m = 7  # input size
p = 4  # output size
h = 20  # hidden dimension (number of REN units)

# Load training data
dataset_csv = pd.read_csv('dataset_xid_train_shuffle.csv', sep=',')
dataset = np.array(dataset_csv.iloc[:, 0:p + m])
X_train = torch.tensor(dataset[:, p:], dtype=torch.float32)
Y_train = torch.tensor(dataset[:, :p], dtype=torch.float32)

# Load validation data
dataset_csv_val = pd.read_csv('dataset_xid_val_shuffle.csv', sep=',')
dataset_val = np.array(dataset_csv_val.iloc[:, 0:p + m])
X_val = torch.tensor(dataset_val[:, p:], dtype=torch.float32)
Y_val = torch.tensor(dataset_val[:, :p], dtype=torch.float32)

# === REN Model ===
class ExplicitREN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.B = nn.Linear(input_dim, hidden_dim, bias=False)      # B: input-to-state
        self.Bs_full = nn.Parameter(torch.empty(hidden_dim, hidden_dim))  # Bs: strictly lower triangular recurrence
        self.Ds = nn.Linear(hidden_dim, output_dim, bias=False)    # Ds: state-to-output
        self.D = nn.Linear(input_dim, output_dim, bias=False)      # D: direct input-to-output
        self.activation = torch.tanh

        # Initialize and apply strict lower triangular mask to Bs
        nn.init.xavier_uniform_(self.Bs_full)
        tril_mask = torch.tril(torch.ones_like(self.Bs_full), diagonal=-1)
        self.register_buffer("tril_mask", tril_mask)

    def forward(self, u):
        """
        Implements an explicit REN with:
            s_i = tanh( ∑_{j < i} Bs_{ij} s_j + (B u)_i )
            y = Ds s + D u

        where Bs is strictly lower triangular.
        """
        batch_size = u.shape[0]
        hidden_dim = self.Bs_full.shape[0]

        Bs = self.Bs_full * self.tril_mask
        Bu = self.B(u)  # shape: (batch_size, hidden_dim)

        s_list = []
        for i in range(hidden_dim):
            if i == 0:
                s_i = self.activation(Bu[:, i])
            else:
                s_prev = torch.stack(s_list, dim=1)
                bs_row = Bs[i, :i]
                bsz = torch.matmul(s_prev, bs_row.T)
                s_i = self.activation(bsz + Bu[:, i])
            s_list.append(s_i)

        s = torch.stack(s_list, dim=1)  # shape: (batch_size, hidden_dim)
        y = self.Ds(s) + self.D(u)      # final output
        return y



# Instantiate model and training components
model = ExplicitREN(m, h, p)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=128)

# Training loop
train_losses, val_losses = [], []
for epoch in range(2000):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")

# === Plots ===
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs (REN)")
plt.legend()
plt.grid(True)
plt.show()

# === Prediction vs Ground Truth ===
model.eval()
with torch.no_grad():
    Y_pred_val = model(X_val).numpy()
    Y_true_val = Y_val.numpy()

plt.figure(figsize=(10, 8))
for i in range(p):
    plt.subplot(p, 1, i + 1)
    plt.plot(Y_true_val[1:300, i], label='True')
    plt.plot(Y_pred_val[1:300, i], '--', label='REN Prediction')
    plt.ylabel(f'Output {i + 1}')
    if i == 0:
        plt.title("REN Fit on Validation Set")
    if i == p - 1:
        plt.xlabel("Sample Index")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
