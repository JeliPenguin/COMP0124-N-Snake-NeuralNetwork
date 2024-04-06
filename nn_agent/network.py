import torch
import torch.nn as nn 
from util import device  # Assuming 'device' is a custom utility for managing device placement.

class IQLNet(nn.Module):
    """Neural network architecture for Independent Q-Learning (IQL) agent."""
    def __init__(self, n_observations, n_actions):
        """
        Initializes the IQL network architecture.
        
        Args:
            n_observations (int): Number of observations.
            n_actions (int): Number of actions.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, state):
        """
        Defines the forward pass of the network.
        
        Args:
            state (torch.Tensor): Input state tensor.
        
        Returns:
            torch.Tensor: Output action tensor.
        """
        return self.model(state.to(device))

    
class MADDPGActorNet(nn.Module):
    """Actor neural network architecture for MADDPG agent."""
    def __init__(self, n_observations, n_actions):
        """
        Initializes the MADDPG actor network architecture.
        
        Args:
            n_observations (int): Number of observations.
            n_actions (int): Number of actions.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_observations, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, state: torch.Tensor):
        """
        Defines the forward pass of the network.
        
        Args:
            state (torch.Tensor): Input state tensor.
        
        Returns:
            torch.Tensor: Output action tensor.
        """
        return self.model(state.to(device))


class MADDPGCriticNet(nn.Module):
    """Critic neural network architecture for MADDPG agent."""
    def __init__(self, n_observations, n_actions, n_agents):
        """
        Initializes the MADDPG critic network architecture.
        
        Args:
            n_observations (int): Number of observations.
            n_actions (int): Number of actions.
            n_agents (int): Number of agents.
        """
        super().__init__()

        self.n_obs = n_observations
        self.n_act = n_actions
        self.n_age = n_agents

        self.model = nn.Sequential(
            nn.Linear(n_observations + n_agents * n_actions, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Defines the forward pass of the network.
        
        Args:
            state (torch.Tensor): Input state tensor.
            action (torch.Tensor): Input action tensor.
        
        Returns:
            torch.Tensor: Output value tensor.
        """
        x = torch.cat([state.to(device), action.to(device)], dim=1)
        return self.model(x)
    
# Placeholder classes for PPO agent architectures
class IndependentPPOActor(nn.Module):
    """Placeholder for Independent PPO actor network."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class IndependentPPOCritic(nn.Module):
    """Placeholder for Independent PPO critic network."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CentralisedPPOActor(nn.Module):
    """Placeholder for Centralised PPO actor network."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CentralisedPPOCritic(nn.Module):
    """Placeholder for Centralised PPO critic network."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
