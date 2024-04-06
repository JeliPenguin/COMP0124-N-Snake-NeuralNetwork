import torch
from replay import ReplayMemory, Transition
from agent import MADDPGAgent
import matplotlib.pyplot as plt
from util import onehot_from_logits

class MAOptimizer:
    """Base class for Multi-Agent Optimizers."""
    def __init__(self, agents):
        """
        Initializes the optimizer with agents.
        
        Args:
            agents (list): List of agents.
        """
        self.agents = agents

    def optimize(self):
        """Abstract method for optimization."""
        raise NotImplementedError
    
    def save(self):
        """Saves policies of all agents."""
        for agent in self.agents:
            agent.save()

    def optimize(self, state_tensor: torch.Tensor, rewards, next_state_tensor: torch.Tensor, total_steps, episode):
        """Abstract method for optimization."""
        raise NotImplementedError

class IQLOptimizer(MAOptimizer):
    """Optimizer for Independent Q-Learning (IQL) agents."""
    def __init__(self, agents, batch_size):
        """
        Initializes the optimizer for IQL agents.
        
        Args:
            agents (list): List of IQL agents.
            batch_size (int): Batch size for replay memory.
        """
        super().__init__(agents)

        # Create replay memory for each agent
        self.memories = [ReplayMemory(50000, batch_size), ReplayMemory(50000, batch_size)]
    
    def memorize(self, memory, state_tensor: torch.Tensor, reward: int, next_state_tensor: torch.Tensor, action_tensor):
        """Stores a transition in the replay memory."""
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
    
    def optimize(self, state_tensor: torch.Tensor, rewards, next_state_tensor: torch.Tensor, total_steps, episode):
        """Optimizes the IQL agents."""
        for i, agent in enumerate(self.agents):
            self.memorize(self.memories[i], state_tensor, rewards[i], next_state_tensor, agent.action_tensor)
            agent.optimize(self.memories[i])

    def show_loss(self):
        """Plots the actor loss."""
        plt.plot(self.agents[0].policy.loss)
        plt.plot(self.agents[1].policy.loss)
        plt.title("Agent Losses")
        plt.show()

class MADDPGOptimizer(MAOptimizer):
    """Centralised Optimizer for Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agents."""
    def __init__(self, agents, batch_size, exploit_eps):
        """
        Initializes the optimizer for MADDPG agents.
        
        Args:
            agents (list): List of MADDPG agents.
            batch_size (int): Batch size for replay memory.
            exploit_eps (int): Number of episodes before starting exploitation.
        """
        super().__init__(agents)
        self.memory = ReplayMemory(50000, batch_size)
        self.exploit_eps = exploit_eps

    def memorize(self, state_tensor: torch.Tensor, rewards: list[int], next_state_tensor: torch.Tensor, all_actions_tensor: torch.Tensor):
        """Stores a transition in the replay memory."""
        all_rewards_tensor = torch.tensor([rewards], dtype=torch.float32)
        self.memory.push(state_tensor, all_actions_tensor, next_state_tensor, all_rewards_tensor)
    
    def optimize(self, state_tensor: torch.Tensor, rewards, next_state_tensor: torch.Tensor, total_steps, episode):
        """Optimizes the MADDPG agents."""
        agent1_action = self.agents[0].get_current_action()
        agent2_action = self.agents[1].get_current_action()
        all_actions_tensor = torch.cat((agent1_action, agent2_action), dim=1)

        self.memorize(state_tensor, rewards, next_state_tensor, all_actions_tensor)

        if self.memory.ready() and total_steps % 10 == 0:
            transitions = self.memory.sample()
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state)
            actions_batch = torch.cat(batch.action)
            rewards_batch = torch.cat(batch.reward)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            next_state_batch = non_final_next_states

            all_agents_target_next_actions = []
            all_agents_next_actions = []

            for agent_idx, agent in enumerate(self.agents):
                agent.target_next_action = agent.policy.target_actor_net.forward(next_state_batch)
                agent.target_next_action = onehot_from_logits(agent.target_next_action)
                all_agents_target_next_actions.append(agent.target_next_action)
                next_action = agent.policy.actor_net.forward(state_batch)
                next_action = onehot_from_logits(next_action)
                all_agents_next_actions.append(next_action)

            for agent_idx, agent in enumerate(self.agents):
                batch = [state_batch, next_state_batch, rewards_batch[:, agent_idx], all_agents_target_next_actions, all_agents_next_actions, actions_batch, non_final_mask]
                agent.optimize(batch)
        
                if total_steps % 20 == 0:
                    agent.policy.soft_update_network()

    def show_loss(self):
        """Plots actor and critic losses."""
        plt.plot(self.agents[0].policy.actor_loss_history)
        plt.plot(self.agents[1].policy.actor_loss_history)
        plt.title("Actor Loss")
        plt.show()

        plt.plot(self.agents[0].policy.critic_loss_history)
        plt.plot(self.agents[1].policy.critic_loss_history)
        plt.title("Critic Loss")
        plt.show()
