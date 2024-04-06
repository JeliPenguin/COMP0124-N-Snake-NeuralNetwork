from env import *
from agent import *
from tqdm import tqdm
from util import *


test_runs = 100

def test(agent_type,game):
    if agent_type == "IQL":
        print("Testing IQL AGENTS")
        save = SAVEPATH + "/tests/IQL"
    else:
        print("Testing MADDPG AGENTS")
        save = SAVEPATH + "/tests/MADDPG"

    for eps in tqdm(range(test_runs)):
        done = False

        game.reset()

        agent1_reward = 0
        agent2_reward = 0

        
        while not done:
            done,reward,_,_ = game.step()


            agent1_reward += reward[0]
            agent2_reward += reward[1]

        game.render_game(save+f"/testrun{eps}.gif")


def IQL_test(game):
    n_observations = game.get_n_observations()
    agent1 = IQLAgent(1,n_observations,IQL_PARAMS,load_save=True)
    agent2 = IQLAgent(2,n_observations,IQL_PARAMS,load_save=True)
    game.set_agents(agent1,agent2)

    test("IQL",game)

def MADDPG_test(game):
    n_observations = game.get_n_observations()
    agent1 = MADDPGAgent(1,n_observations,MADDPG_PARAMS,2,load_save=True)
    agent2 = MADDPGAgent(2,n_observations,MADDPG_PARAMS,2,load_save=True)
    game.set_agents(agent1,agent2)

    test("MADDPG",game)

if __name__ == "__main__":
    game = SnakeGame(GAME_SIZE)

    IQL_test(game)
    # MADDPG_test(game)

    