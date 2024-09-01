import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import get_model, train_model

# Dummy DataLoader - replace with actual data loading and preprocessing
train_data = np.random.rand(100, 3)
train_labels = np.random.rand(100, 2)
train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                              torch.tensor(train_labels, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Model initialization
model = get_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
trained_model = train_model(model, train_loader, criterion, optimizer, epochs=10)
torch.save(trained_model.state_dict(), 'construction_model.pth')
