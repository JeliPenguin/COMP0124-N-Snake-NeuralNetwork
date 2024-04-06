import torch
import torch.functional as F
import torch.nn as nn 
from util import device


    
class IQLNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state):
        return self.model(state.to(device))

    
class MADDPGActorNet(nn.Module):
    def __init__(self,n_observations,n_actions) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_observations,500),
            nn.ReLU(),
            nn.Linear(500,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,n_actions)   
        )

    def forward(self,state:torch.Tensor):
        return self.model(state.to(device))


class MADDPGCriticNet(nn.Module):
    def __init__(self, n_observations,n_actions,n_agents) -> None:
        super().__init__()

        self.n_obs = n_observations
        self.n_act = n_actions
        self.n_age = n_agents

        self.model = nn.Sequential(
            nn.Linear(n_observations+n_agents*n_actions,500),
            nn.ReLU(),
            nn.Linear(500,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,state:torch.Tensor,action:torch.Tensor):
        x = torch.cat([state.to(device), action.to(device)], dim=1)
        return self.model(x)
    
class IndependentPPOActor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class IndependentPPOCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CentralisedPPOActor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class CentralisedPPOCritic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)