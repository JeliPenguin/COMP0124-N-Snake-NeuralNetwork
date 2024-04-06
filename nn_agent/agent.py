import random
from env import Move
import numpy as np
from policy import *
from util import *


class BaseNNAgent():
    def __init__(self, id:int):
        """
        Base class for neural network-based agents.
        
        Args:
            id (int): Unique identifier for the agent.
        """
        self.id = id
        self.action_space = [Move.LEFT,Move.RIGHT,Move.UP,Move.DOWN]
        self.state_tensor = None
        self.action_tensor = None
        self.policy = None
    
    def get_current_action(self) -> torch.Tensor:
        """
        Retrieves the current action tensor.
        
        Returns:
            torch.Tensor: Current action tensor.
        """
        return self.action_tensor
    

class IQLAgent(BaseNNAgent):
    def __init__(self, id: int, n_observations: int, hyper_params:dict,load_save=False) -> None:
        """
        Independent Q-Learning (IQL) agent class.
        
        Args:
            id (int): Unique identifier for the agent.
            n_observations (int): Number of observations.
            hyper_params (dict): Dictionary containing hyperparameters.
            load_save (bool): Flag to load/save the model parameters.
        """
        super().__init__(id)
        self.policy = IQLPolicy(n_observations,self.action_space,hyper_params,load_save,id)
        
        
    def optimize(self,replay_buffer):
        """
        Optimizes the agent's policy using experience replay buffer.
        
        Args:
            replay_buffer: Replay buffer containing experiences.
        """
        self.policy.optimize(replay_buffer)
    
    def save(self):
        """Saves the policy."""
        self.policy.save_policy()

    def step(self, board_state: np.ndarray, pos: tuple[int, int]) -> Move:
        """
        Performs a step/action based on the given board state and position.
        
        Args:
            board_state (np.ndarray): Current state of the game board.
            pos (tuple[int, int]): Current position of the agent on the board.
        
        Returns:
            Move: Action to take.
        """
        self.state_tensor = state_encoder(board_state,pos)
        self.action_tensor = self.policy.choose_action(self.state_tensor)
        self.action_tensor = self.action_tensor.to(torch.int64)
        return self.action_space[int(self.action_tensor.item())]

class MADDPGAgent(BaseNNAgent):
    def __init__(self, id: int, n_observations: int, hyper_params:dict,n_agents:int,load_save=False):
        """
        Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent class.
        
        Args:
            id (int): Unique identifier for the agent.
            n_observations (int): Number of observations.
            hyper_params (dict): Dictionary containing hyperparameters.
            n_agents (int): Total number of agents in the environment.
            load_save (bool): Flag to load/save the model parameters.
        """
        super().__init__(id)
        self.load_save = load_save
        self.policy = MADDPGPolicy(n_observations,self.action_space,hyper_params,n_agents,load_save,id)

    def optimize(self,batch):
        """
        Optimizes the agent's policy using a batch of experiences.
        
        Args:
            batch: Batch of experiences.
        """
        self.policy.optimize(batch)

    def save(self):
        """Saves the policy."""
        self.policy.save_policy()
    
    def exploit(self):
        """Stops exploration noise."""
        self.policy.stop_noise()

    def step(self, board_state: np.ndarray, pos: tuple[int, int]) -> Move:
        """
        Performs a step/action based on the given board state and position.
        
        Args:
            board_state (np.ndarray): Current state of the game board.
            pos (tuple[int, int]): Current position of the agent on the board.
        
        Returns:
            Move: Action to take.
        """
        self.state_tensor = state_encoder(board_state,pos)
        action_logit = self.policy.choose_action(self.state_tensor)

        if self.load_save:
            self.action_tensor = gumbel_softmax(action_logit,hard=True)
        else:
            self.action_tensor = gumbel_softmax(action_logit,hard=True)
        action_index = torch.argmax(self.action_tensor).view(1,1)
        return self.action_space[int(action_index.item())]

class MAPPOAgent(BaseNNAgent):
    def __init__(self, id: int):
        """
        Multi-Agent Proximal Policy Optimization (MAPPO) agent class.
        
        Args:
            id (int): Unique identifier for the agent.
        """
        super().__init__(id)

    def optimise(self,done,reward,s_prime):
        """
        Optimizes the agent's policy.
        
        Args:
            done: Whether the episode is done.
            reward: The reward received.
            s_prime: Next state.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
    
    
class CentralisedMAPPOAgent(MAPPOAgent):
    def __init__(self, id: int):
        """
        Centralised version of Multi-Agent Proximal Policy Optimization (MAPPO) agent class.
        
        Args:
            id (int): Unique identifier for the agent.
        """
        super().__init__(id)
        self.policy = CentralisedMAPPOPolicy()
    
class IndependentMAPPOAgent(MAPPOAgent):
    def __init__(self, id: int):
        """
        Independent version of Multi-Agent Proximal Policy Optimization (MAPPO) agent class.
        
        Args:
            id (int): Unique identifier for the agent.
        """
        super().__init__(id)
        self.policy = IndependentMAPPOPolicy()
