import random
from env import Move
import numpy as np
from policy import *
from util import *


class BaseNNAgent():
    def __init__(self, id:int):
        self.id = id
        self.action_space = [Move.LEFT,Move.RIGHT,Move.UP,Move.DOWN]
        self.state_tensor = None
        self.action_tensor = None
        self.policy = None
    
    def get_current_action(self) -> torch.Tensor:
        return self.action_tensor
    

class IQLAgent(BaseNNAgent):
    def __init__(self, id: int, n_observations: int, hyper_params:dict,load_save=False) -> None:
        super().__init__(id)
        self.policy = IQLPolicy(n_observations,self.action_space,hyper_params,load_save,id)
        
        
    def optimize(self,replay_buffer):
        self.policy.optimize(replay_buffer)
    
    def save(self):
        self.policy.save_policy()

    def step(self, board_state: np.ndarray, pos: tuple[int, int]) -> Move:
        self.state_tensor = state_encoder(board_state,pos)
        self.action_tensor = self.policy.choose_action(self.state_tensor)
        self.action_tensor = self.action_tensor.to(torch.int64)
        return self.action_space[int(self.action_tensor.item())]

class MADDPGAgent(BaseNNAgent):
    def __init__(self, id: int, n_observations: int, hyper_params:dict,n_agents:int,load_save=False):
        super().__init__(id)
        self.load_save = load_save
        self.policy = MADDPGPolicy(n_observations,self.action_space,hyper_params,n_agents,load_save,id)

    def optimize(self,batch):
        self.policy.optimize(batch)

    def save(self):
        self.policy.save_policy()
    
    def exploit(self):
        self.policy.stop_noise()

    def step(self, board_state: np.ndarray, pos: tuple[int, int]) -> Move:
        self.state_tensor = state_encoder(board_state,pos)
        action_logit = self.policy.choose_action(self.state_tensor)

        if self.load_save:
            self.action_tensor = gumbel_softmax(action_logit,hard=True)
        else:
            self.action_tensor = gumbel_softmax(action_logit,hard=True)
        # print(self.action_tensor)
        action_index = torch.argmax(self.action_tensor).view(1,1)
        return self.action_space[int(action_index.item())]

    
class MAPPOAgent(BaseNNAgent):
    def __init__(self, id: int):
        super().__init__(id)

    def optimise(self,done,reward,s_prime):
        raise NotImplementedError
    
    
class CentralisedMAPPOAgent(MAPPOAgent):
    def __init__(self, id: int):
        super().__init__(id)
        self.policy = CentralisedMAPPOPolicy()
    
class IndependentMAPPOAgent(MAPPOAgent):
    def __init__(self, id: int):
        super().__init__(id)
        self.policy = IndependentMAPPOPolicy()