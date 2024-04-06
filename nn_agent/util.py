import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SAVEPATH = PATH+"/nn_agent/save"
GAME_SIZE = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)

IQL_PARAMS = {
    "batch_size": 1024,
    "gamma": 0.9,
    "eps_start": 0.99,
    "eps_end": 0.1,
    "eps_decay": 12000,
    "tau": 0.005,
    "lr": 0.001,
}

MADDPG_PARAMS = {
    "batch_size": 1024,
    "gamma": 0.9,
    "eps_start": 0,
    "eps_end": 0,
    "eps_decay": 0,
    "tau": 0.005,
    "lr": 0.001,
}

episodes = 20000

def state_encoder(board_state: np.ndarray, pos) -> torch.Tensor:
        # return torch.reshape(torch.cat((torch.flatten(torch.tensor(board_state)),torch.flatten(torch.tensor(pos))),0),(1,-1)).to(torch.float32)
    return torch.reshape(torch.flatten(torch.tensor(board_state)),(1,-1)).to(torch.float32)



# Following functions were taken from https://github.com/shariqiqbal2810/maddpg-pytorch
def onehot_from_logits(logits, eps=0.0):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps).to(device)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y