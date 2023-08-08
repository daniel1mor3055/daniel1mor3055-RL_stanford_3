import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
import torch.nn.functional as F


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, etc.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate
        observation_dim = self.env.observation_space.shape[0]

        # Create the neural network baseline
        self.network = build_mlp(
            input_size=observation_dim,
            output_size=1,
            n_layers=self.config.n_layers,
            size=self.config.layer_size
        )

        # Define the optimizer for the baseline
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        # Pass the observations through the network
        output = self.network(observations)
        output = output.squeeze()

        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)
        # Use the forward pass of the baseline network to get the predicted value of the state
        predicted_values = self(observations)

        # Compute the advantage estimates
        advantages = returns - predicted_values.detach().numpy()

        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer once.

        You may find the following documentation useful:
        https://pytorch.org/docs/stable/nn.functional.html
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        # Zero the gradients of the optimizer
        self.optimizer.zero_grad()

        # Forward pass to get the predicted values
        predicted_values = self(observations)

        # Compute the Mean Squared Error loss
        loss = F.mse_loss(predicted_values, returns)

        # Backpropagate the loss
        loss.backward()

        # Update the weights of the network
        self.optimizer.step()
