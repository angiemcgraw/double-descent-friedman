"""
Angie McGraw
Last updated: March 31st, 2026

- Complexity scales the number of hidden units (number of layers, optionally).
- Training loop monitors MSE
- Adjustable epochs and learning rate

Features
1. Single hidden layer MLP
- Literature often uses simple MLPs to illustrate the interpolation peak.
- Single-layer keeps the network comparable to the other models, especially in terms
of the number of parameters.

2. Parameter control via complexity.
- complexity controls the number of hidden units.

3. He initialization with ReLU
- This is standard initialization for MLPs with ReLU.
- Prevents vanishing/exploding gradients.

4. SGD with momentum
- Nakkiran et al.
- Adam would adapt learning rates and flatten the peak.

5. Label corruption handled in run_seed
"""

import torch
from torch import  nn
from models.base_model import BaseModel

class NeuralNetwork(BaseModel):
    def __init__(self, input_dim, complexity=10, lr=0.01, epochs=200, 
                 n_layers=1, optimizer="adam"):
        """
        input_dim: number of input features
        complexity: number of hidden units per layer
        n_layers: number of hidden layers (default = 1)
        - ReLU activation 
        - He init - for ReLU
        - Full-batch gradient descent
        - Label corruption handled in run_seed
        """
        super().__init__(complexity)
        
        self.input_dim = input_dim
        self.hidden_dim = complexity
        self.lr = lr
        self.epochs = epochs
        self.n_layers = n_layers
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Build MLP
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))     # output layer
        self.model = nn.Sequential(*layers).to(self.device)

        # He init
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

        self.loss_fn = nn.MSELoss()

        # Optimizer selection
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr
            )
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=0.0
            )

    def fit(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Full-batch gradient descent
        for epoch in range(self.epochs):
            pred = self.model(X_tensor)
            loss = self.loss_fn(pred, y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Diagnostic
        with torch.no_grad():
            final_train = self.loss_fn(self.model(X_tensor), y_tensor).item()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"     h={self.hidden_dim:4d} | params={n_params:5d} | train_loss={final_train:.6f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(X_tensor).cpu().numpy().flatten()