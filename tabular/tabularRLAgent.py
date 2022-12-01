import gym
import random
import numpy
# class TabularRLAgent(object):
#     def __init__(self, states, actions, q_func=None):

def greedy_act(actions):
    """
    Greedy action selection based on a q or value function
    """
    best_q_val = -np.inf
    best_act = None
    for action in actions:
        state_action_pair = (state,action)

        if self.q_func[state_action_pair] > best_q_val:
            best_q_val = self.q_func[state_action_pair]
            best_act = action

    return action

def e_greedy_act(actions, e):
    """
    Given a list of actions to take, returns one with an e-greedy probability.
    """
    rand = random.random()

    num_actions = len(actions)

    #find best action