import numpy as np
from enum import Enum
from array2gif import write_gif

class Move(Enum):

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Result(Enum):

    TIE = 0
    WIN1 = 1
    WIN2 = 2
    SELFKILL1 = 3
    SELFKILL2 = 4
    BOTHSELFKILL = 5
    


class SnakeGame:

    def __init__(self, n=9):
        self.n = n
        self.reset()
        

    def reset(self):
        self.state = -np.ones((self.n+2, self.n+2))
        self.state[1:self.n+1, 1:self.n+1] = np.zeros((self.n, self.n))
        
        self.history = [self.state.copy()]

        self.pos1 = (self.n//4 + 1, self.n//2 + 1)
        self.pos2 = (1 + (3 * self.n // 4), self.n//2 + 1)

        self.add_frame()

        return self.state.copy(),[self.pos1,self.pos2]

    def get_n_observations(self):
        return np.prod(self.state.shape)
    
    def set_agents(self,agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2

    def terminated(self):
        x1, y1 = self.pos1
        x2, y2 = self.pos2

        if (x1 == x2) and (y1 == y2):
            return Result.TIE
        elif self.state[x1, y1] == self.agent1.id and self.state[x2, y2] == self.agent2.id:
            return Result.BOTHSELFKILL
        elif self.state[x1, y1] == -1 and self.state[x2, y2] == -1:
            return Result.TIE
        elif self.state[x1, y1] == self.agent1.id:
            return Result.SELFKILL1
        elif self.state[x2,y2] == self.agent2.id:
            return Result.SELFKILL2
        elif self.state[x1, y1] != 0:
            return Result.WIN2
        elif self.state[x2, y2] != 0:
            return Result.WIN1
        
        return False


    def reward_function(self,done):
        if done == Result.WIN1:
            return [0.1,-0.1]
        elif done == Result.WIN2:
            return [-0.1,0.1]
        elif done == Result.TIE:
            return [-0.1,-0.1]
        elif done == Result.SELFKILL1:
            return [-0.5,0.1]
        elif done == Result.SELFKILL2:
            return [0.1,-0.5]
        elif done == Result.BOTHSELFKILL:
            return [-0.5,-0.5]
        return [0.05,0.05]

    def step(self):

        pos = [self.pos1,self.pos2]

        m1 = self.agent1.step(self.state, pos)
        m2 = self.agent2.step(self.state, pos)

        x1, y1 = self.pos1
        x2, y2 = self.pos2

        self.state[x1, y1] = self.agent1.id
        self.state[x2, y2] = self.agent2.id

        self.pos1 = self.move_update(self.pos1, m1)
        self.pos2 = self.move_update(self.pos2, m2)

        self.add_frame()

        done = self.terminated()
        reward = self.reward_function(done)

        # Assumes full observability of the environment
        if done:
            s_prime = None
        else:
            s_prime = self.state.copy()

        pos_ = [self.pos1,self.pos2]

        return done,reward,s_prime,pos_


    def play(self, max_steps = 100000000):

        done = False

        while not done:

            max_steps -= 1
            if max_steps <= 0:
                raise Exception("Game Failed to terminate in max steps.")

            done,_,_,_= self.step()
        
        print(done)


    def move_update(self, pos:tuple[int, int], move:Move) -> tuple[int, int]:
        x, y = pos
        if move == Move.UP:
            return (x, y+1)
        if move == Move.DOWN:
            return (x, y-1)
        if move == Move.LEFT:
            return (x-1, y)
        if move == Move.RIGHT:
            return (x+1, y)
        
    def add_frame(self):

        frame = self.state.copy()
        x1, y1 = self.pos1
        x2, y2 = self.pos2

        frame[x1, y1] = -1
        frame[x2, y2] = -1

        self.history.append(frame)

    def render_game(self, filepath:str):

        dataset = []

        for state in self.history:
            r_grid = state.copy()
            g_grid = state.copy()
            b_grid = state.copy()

            # set wall color
            r_grid[state == -1] = 255
            g_grid[state == -1] = 255
            b_grid[state == -1] = 255

            # set agent 1 color
            r_grid[state == self.agent1.id] = 255
            g_grid[state == self.agent1.id] = 0
            b_grid[state == self.agent1.id] = 0

            # set agent 2 color
            r_grid[state == self.agent2.id] = 0
            g_grid[state == self.agent2.id] = 0
            b_grid[state == self.agent2.id] = 255

            dataset.append(np.stack([r_grid, g_grid, b_grid], axis=0))

        write_gif(dataset, filepath, fps=15)