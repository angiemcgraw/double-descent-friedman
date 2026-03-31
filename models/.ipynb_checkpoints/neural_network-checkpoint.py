"""
Angie McGraw
Last updated: March 31st, 2026

- Complexity scales the number of hidden units (number of layers, optionally).
- Training loop monitors MSE
- Adjustable epochs and learning rate
"""

import torch
from torch import  nn
from torch.utils.data import TensorDataset, DataLoader
from models.base_model import BaseModel

class NeuralNetwork(BaseModel):
    def __init__(self, input_dim, complexity=10, lr=1e-3, epochs=500, batch_size=32, n_layers=None):
        """
        input_dim: number of input features
        complexity: number of hidden units per layer
        n_layers: number of hidden layers (default = 2)
        """
        super().__init__(complexity)
        self.input_dim = input_dim
        self.hidden_dim = complexity
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_layers = n_layers or 3      # default 3 hidden layers

        # Build MLP dynamically
        layers = []
        in_dim = self.input_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))     # output layers

        # Define simple MLP 
        self.model = nn.Sequential(*layers).to(self.device)

        # Apply the He initialization to all Linear layers
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_tensor).cpu().numpy().flatten()