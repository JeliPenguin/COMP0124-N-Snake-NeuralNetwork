from network import *
from torch import Tensor, optim
import torch.nn.functional as F
from replay import *
import numpy as np
from util import *


class NNPolicy():
    def __init__(self,n_observations:int , action_space:list, hyper_params:dict,load_save) -> None:
        self.n_observations = n_observations
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.hyper_params = hyper_params
        self.lr = hyper_params["lr"]
        self.batch_size = hyper_params["batch_size"]
        self.gamma = hyper_params["gamma"]
        self.eps_start = hyper_params["eps_start"]
        self.eps_end = hyper_params["eps_end"]
        self.eps_decay = hyper_params["eps_decay"]
        self.tau = hyper_params["tau"]
        self.load_save = load_save

    def memorize(self):
        raise NotImplementedError

    def choose_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def choose_random_action(self) -> torch.Tensor:
        rand_action = np.random.randint(0, self.n_actions)
        return torch.tensor([[rand_action]], device=device)

    def choose_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save_policy(self,policy_save_path):
        raise NotImplementedError
    
    def hard_update(self):
        pass

    def soft_update(self,target_net,behaviour_net):
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = behaviour_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key]*(1-self.tau)
        target_net.load_state_dict(target_net_state_dict)
    
    def hard_update(self,target_net,behaviour_net):
        target_net.load_state_dict(behaviour_net.state_dict())


class IQLPolicy(NNPolicy):
    def __init__(self,n_observations:int , action_space:list, hyper_params:dict,load_save,id) -> None:
        super().__init__(n_observations,action_space,hyper_params,load_save)

        self.policy_save_path = SAVEPATH+"/train/IQL/IQL" + str(id) + ".pt"
        self.loss_save_path = SAVEPATH+"/train/IQL/IQL" + str(id) + "_loss.pt"

        if self.load_save:
            print("LOADING SAVE")
            policy_param = torch.load(self.policy_save_path)
            self.policy_net = IQLNet(n_observations, self.n_actions).to(device)
            self.policy_net.load_state_dict(policy_param)
            
        else:
            self.policy_net = IQLNet(n_observations, self.n_actions).to(device)
            self.target_net = IQLNet(n_observations, self.n_actions).to(device)
            self.hard_update(self.target_net,self.policy_net)

        self.eps_done = 0

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.loss = []
    
              
    def optimize(self,replay_buffer):
        """Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""
        if replay_buffer.ready():
        
            transitions = replay_buffer.sample()
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None]).to(device)
            
            state_batch = torch.cat(batch.state).to(device)
            action_batch = torch.cat(batch.action).to(device)
            reward_batch = torch.cat(batch.reward).to(device)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # print(state_batch.shape)
            state_action_values = self.policy_net(
                state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = nn.MSELoss()
            loss = criterion(state_action_values,
                            expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            self.loss.append(loss.detach().item())
            
            self.soft_update(self.target_net,self.policy_net)

    def choose_greedy_action(self,state_tensor:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_net(state_tensor.to(device)).max(1)[1].view(1, 1)
        
    def epsilon_decay_schedule(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.eps_done / self.eps_decay)

    def choose_action(self,state_tensor:torch.Tensor) -> torch.Tensor:
        # Ordinary Epsilon greedy
        p = np.random.random()
        if self.load_save:
            eps_thresh = self.eps_end
        else:
            eps_thresh = self.epsilon_decay_schedule()
        if p > eps_thresh:
            return self.choose_greedy_action(state_tensor)
        return self.choose_random_action()
    
    def save_policy(self):
        torch.save(self.policy_net.state_dict(), self.policy_save_path)
        torch.save(self.loss,self.loss_save_path)


class MADDPGPolicy(NNPolicy):
    def __init__(self, n_observations: int, action_space: list, hyper_params: dict,n_agents:int,load_save,id) -> None:
        super().__init__(n_observations, action_space, hyper_params,load_save)
        self.add_noise = True
        self.policy_save_path = SAVEPATH+"/train/MADDPG/MADDPG" + str(id)
        self.actor_loss_history = []
        self.critic_loss_history = []

        if self.load_save:
            print("LOADING SAVE")
            self.actor_net = MADDPGActorNet(n_observations,self.n_actions).to(device)
            actor_params = torch.load(self.policy_save_path+"_actor.pt")
            self.actor_net.load_state_dict(actor_params)
        else:

            self.actor_net = MADDPGActorNet(n_observations,self.n_actions).to(device)
            self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)

            self.target_actor_net = MADDPGActorNet(n_observations,self.n_actions).to(device)
            self.target_actor_optimizer = optim.Adam(self.target_actor_net.parameters(), lr=self.lr)

            self.critic_net = MADDPGCriticNet(n_observations,self.n_actions,n_agents).to(device)
            self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

            self.target_critic_net = MADDPGCriticNet(n_observations,self.n_actions,n_agents).to(device)
            self.target_critic_optimizer = optim.Adam(self.target_critic_net.parameters(),lr=self.lr)

            self.hard_update(self.target_actor_net,self.actor_net)
            self.hard_update(self.target_critic_net,self.critic_net)

    def stop_noise(self):
        self.add_noise = False

    def soft_update_network(self):
        # Soft update actor and critic target network's weights
        self.soft_update(self.target_actor_net,self.actor_net)
        self.soft_update(self.target_critic_net,self.critic_net)

    def choose_action(self, state_tensor: Tensor) -> Tensor:
        # No epsilon greedy in MADDPG, add noise to action for exploration
        actions = self.actor_net.forward(state_tensor)
        # print(actions)
        if not self.load_save and self.add_noise:
            noise = torch.rand(self.n_actions,device=device)
            actions = actions + noise
        # print(actions)

        return actions.detach()
    
    def optimize(self,batch):
        state_batch,next_state_batch,rewards_batch,all_agents_target_next_actions,all_agents_next_actions,actions_batch,not_final_mask = batch

        target_next_actions = torch.cat([acts for acts in all_agents_target_next_actions], dim=1)
        next_actions = torch.cat([acts for acts in all_agents_next_actions], dim=1)
        
        # target_critic_value[dones[:,0]] = 0.0
        critic_value = self.critic_net.forward(state_batch, actions_batch).flatten().to(device)

        provisional_target_critic_value = self.target_critic_net.forward(next_state_batch, target_next_actions).flatten().to(device)
        target_critic_value = torch.zeros_like(rewards_batch).to(device)
        target_critic_value[not_final_mask] = provisional_target_critic_value[:torch.sum(not_final_mask)].to(device)
        
        # print(not_final_mask)
        # print(target_critic_value)

        # assert(rewards_batch.shape == target_critic_value.shape)
        # print("\nReward: ",rewards_batch)
        # print("\nTarget Critic:",target_critic_value)
        # exit()
        target = rewards_batch.to(device) + self.gamma*target_critic_value

        critic_loss = F.mse_loss(target.detach(), critic_value).to(device)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_loss = self.critic_net.forward(state_batch, next_actions).flatten().to(device)
        actor_loss = -torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optimizer.step()

        self.actor_loss_history.append(actor_loss.detach().item())
        self.critic_loss_history.append(critic_loss.detach().item())
    
    def save_policy(self):
        torch.save(self.actor_net.state_dict(), self.policy_save_path+"_actor.pt")
        torch.save(self.critic_net.state_dict(),self.policy_save_path+"_critic.pt")
        torch.save(self.actor_loss_history,self.policy_save_path+"_actor_loss.pt")
        torch.save(self.critic_loss_history,self.policy_save_path+"_critic_loss.pt")

class CentralisedMAPPOPolicy():
    def __init__(self) -> None:
        self.actor_net = CentralisedPPOActor()
        self.critic_net = CentralisedPPOCritic()

class IndependentMAPPOPolicy():
    def __init__(self) -> None:
        self.actor_net = IndependentPPOActor()
        self.critic_net = IndependentPPOCritic()