import torch
from replay import ReplayMemory,Transition
from agent import MADDPGAgent
import matplotlib.pyplot as plt
from util import onehot_from_logits

class MAOptimizer():
    def __init__(self,agents) -> None:
        self.agents = agents

    def optimize(self):
        raise NotImplementedError
    
    def save(self):
        for agent in self.agents:
            agent.save()

    def optimize(self,state_tensor:torch.Tensor,rewards,next_state_tensor:torch.Tensor,total_steps,episode):
        raise NotImplementedError

class IQLOptimizer(MAOptimizer):
    def __init__(self, agents,batch_size) -> None:
        super().__init__(agents)

        self.memories = [ReplayMemory(50000,batch_size),ReplayMemory(50000,batch_size)]
    
    def memorize(self,memory,state_tensor:torch.Tensor,reward:int,next_state_tensor:torch.Tensor,action_tensor):

        reward_tensor = torch.tensor([reward],dtype=torch.float32)

        memory.push(state_tensor, action_tensor,
                          next_state_tensor, reward_tensor)

    
    def optimize(self,state_tensor:torch.Tensor,rewards,next_state_tensor:torch.Tensor,total_steps,episode):
        for i,agent in enumerate(self.agents):
            self.memorize(self.memories[i],state_tensor,rewards[i],next_state_tensor,agent.action_tensor)
            agent.optimize(self.memories[i])

    def show_loss(self):
        plt.plot(self.agents[0].policy.loss)
        plt.plot(self.agents[1].policy.loss)
        plt.title("Actor Loss")
        plt.show()

class MADDPGOptimizer(MAOptimizer):
    # MADDPG uses centralised optimization for all agents
    def __init__(self, agents,batch_size,exploit_eps) -> None:
        super().__init__(agents)
        self.memory = ReplayMemory(50000,batch_size)
        self.exploit_eps = exploit_eps

    def memorize(self, state_tensor:torch.Tensor,rewards:list[int],next_state_tensor:torch.Tensor,all_actions_tensor:torch.Tensor):
        # print("State: ",state_tensor.dtype)
        # print("Action: ",action_tensor.dtype)
        # print("Next State: ",next_state_tensor.dtype)
        # print("Reward: ",reward_tensor.dtype)

        all_rewards_tensor = torch.tensor([rewards],dtype=torch.float32)
        
        self.memory.push(state_tensor, all_actions_tensor,
                          next_state_tensor, all_rewards_tensor)

    
    def optimize(self,state_tensor:torch.Tensor,rewards,next_state_tensor:torch.Tensor,total_steps,episode):
        agent1_action = self.agents[0].get_current_action()
        agent2_action = self.agents[1].get_current_action()
        all_actions_tensor = torch.cat((agent1_action,agent2_action),dim=1)

        # if episode >= self.exploit_eps:
        #     for agent in self.agents:
        #         agent.exploit()

        # print(all_actions_tensor.shape)

        self.memorize(state_tensor,rewards,next_state_tensor,all_actions_tensor)

        if self.memory.ready() and total_steps % 10 == 0:
        
            transitions = self.memory.sample()
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            
            # print(batch.next_state)
            
            state_batch = torch.cat(batch.state)
            actions_batch = torch.cat(batch.action)

            # print("Actions batch shape: ",actions_batch.shape)

            rewards_batch = torch.cat(batch.reward)

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
            next_state_batch = non_final_next_states

            # print(next_state_batch)

            all_agents_target_next_actions = []
            all_agents_next_actions = []
            # old_agents_actions = []

            for agent_idx, agent in enumerate(self.agents):
                agent:MADDPGAgent

                target_next_action = agent.policy.target_actor_net.forward(next_state_batch)

                target_next_action = onehot_from_logits(target_next_action)

                all_agents_target_next_actions.append(target_next_action)

                next_action = agent.policy.actor_net.forward(state_batch)
                next_action = onehot_from_logits(next_action)

                all_agents_next_actions.append(next_action)


            for agent_idx, agent in enumerate(self.agents):
                batch = [state_batch,next_state_batch,rewards_batch[:,agent_idx],all_agents_target_next_actions,all_agents_next_actions,actions_batch,non_final_mask]
                agent.optimize(batch)
        
                if total_steps % 20 == 0:
                    agent.policy.soft_update_network()

    def show_loss(self):
        plt.plot(self.agents[0].policy.actor_loss_history)
        plt.plot(self.agents[1].policy.actor_loss_history)
        plt.title("Actor Loss")
        plt.show()

        plt.plot(self.agents[0].policy.critic_loss_history)
        plt.plot(self.agents[1].policy.critic_loss_history)
        plt.title("Critic Loss")
        plt.show()



