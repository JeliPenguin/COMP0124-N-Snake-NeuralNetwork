from env import SnakeGame
from agent import IQLAgent,MADDPGAgent
from optimizer import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import *
import torch



def train(game:SnakeGame,ma_optimizer:MAOptimizer,episodes:int):
    if isinstance(ma_optimizer,IQLOptimizer):
        print("TRAINING IQL AGENTS")
        save = SAVEPATH + "/train/IQL/IQL"
    else:
        print("TRAINING MADDPG AGENTS")
        save = SAVEPATH + "/train/MADDPG/MADDPG"

    agent1_rewards = []
    agent2_rewards = []

    total_steps = 0

    for eps in tqdm(range(episodes)):
        done = False

        s,pos = game.reset()

        state_tensor = state_encoder(s,pos)

        agent1_reward = 0
        agent2_reward = 0

        steps = 0
        
        while not done:
            done,reward,next_state,next_pos = game.step()

            if next_state is not None:
                next_state_tensor = state_encoder(next_state,next_pos)
            else:
                next_state_tensor = None

            # print("State: ",state_tensor)
            # print("Next State: ",next_state_tensor)

            ma_optimizer.optimize(state_tensor,reward,next_state_tensor,total_steps,eps)

            agent1_reward += reward[0]
            agent2_reward += reward[1]

            steps+=1
            total_steps += 1

            state_tensor = next_state_tensor

        agent1_rewards.append(agent1_reward/steps)
        agent2_rewards.append(agent2_reward/steps)

        if eps%200==0 and eps>0:
            game.render_game(SAVEPATH+"/train/trainrun.gif")
            # ma_optimizer.show_loss()
            # print(done)
            # print("Average Rewards: ",agent1_reward/steps,agent2_reward/steps)

    ma_optimizer.save()
    plt.plot(agent1_rewards)
    plt.plot(agent2_rewards)
    plt.show()

    ma_optimizer.show_loss()
    torch.save(agent1_rewards,save+"_reward_1.pt")
    torch.save(agent1_rewards,save+"_reward_2.pt")

def IQL_train(game):
    n_observations = game.get_n_observations()
    agent1 = IQLAgent(1,n_observations,IQL_PARAMS)
    agent2 = IQLAgent(2,n_observations,IQL_PARAMS)
    game.set_agents(agent1,agent2)

    batch_size = IQL_PARAMS["batch_size"]
    ma_optimizer = IQLOptimizer([agent1,agent2],batch_size)

    train(game,ma_optimizer,episodes)


def MADDPG_train(game):
    n_observations = game.get_n_observations()
    agent1 = MADDPGAgent(1,n_observations,MADDPG_PARAMS,2)
    agent2 = MADDPGAgent(2,n_observations,MADDPG_PARAMS,2)
    game.set_agents(agent1,agent2)

    
    batch_size = MADDPG_PARAMS["batch_size"]
    ma_optimizer = MADDPGOptimizer([agent1,agent2],batch_size,int(0.8*episodes))

    train(game,ma_optimizer,episodes)

if __name__ == "__main__":
    game = SnakeGame(GAME_SIZE)
    IQL_train(game)
    MADDPG_train(game)