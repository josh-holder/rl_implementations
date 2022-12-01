import gym
import numpy as np
import random
from matplotlib import pyplot as plt
import time
from tabularRLAgent import TabularRLAgent

class TabularQLearnAgent(TabularRLAgent):
    def __init__(self, env, epsilon=0.2):
        self.q_func = None
        self.env = env
        self.epsilon = epsilon

    def act_target_policy(self,actions,state):
        """
        Target policy for q-learning is greedy w/r/t the q function.
        """
        best_q_val = -np.inf
        best_act = None
        for action in actions:
            state_action_pair = (state,action)

            if self.q_func[state_action_pair] > best_q_val:
                best_q_val = self.q_func[state_action_pair]
                best_act = action

        return best_act

    def act_behavior_policy(self,actions,state):
        """
        Behavior policy for q-learning is typically e-greedy w/r/t q function.
        (as in this case)
        """

        #find best action
        best_q_val = -np.inf
        best_act = None
        for action in actions:
            state_action_pair = (state,action)

            if self.q_func[state_action_pair] > best_q_val:
                best_q_val = self.q_func[state_action_pair]
                best_act = action

        rand_num = random.random()
        num_actions = len(actions)

        threshold = 0
        old_threshold = 0

        for action in actions:
            if action == best_act:
                threshold += (1-self.epsilon)+self.epsilon/num_actions
            else:
                threshold += self.epsilon/num_actions

            if old_threshold <= rand_num and rand_num <= threshold:
                return action
            else:
                old_threshold=threshold
        
        #If you make it out of this loop, something went wrong
        raise Exception("ERROR: Behavior policy failed to select an action.")

    def trainAgent(self, episodes, alpha=0.9,discount=1):
        """
        Trains the agent for the specified amount of episodes, with the specified
        hyperparameters.

        Returns lists of rewards and lengths for which to use in tracking progress of agents.
        """
        episode = 0

        # init_epsilon = self.epsilon

        rewards = []
        lengths = []
        while episode < episodes:
            # self.epsilon -= init_epsilon/episodes
            observation = self.env.reset()

            terminated = False

            num_states = 0
            total_reward = 0
            traj = []
            while not terminated:
                actions = range(self.env.action_space.n)
                behavior_action = self.act_behavior_policy(actions,observation)
                old_state = observation
                observation, reward, terminated, info = self.env.step(behavior_action)

                greedy_action = self.act_target_policy(actions,observation)
                best_q = self.q_func[(observation,greedy_action)]
                self.q_func[(old_state,behavior_action)] += alpha*(reward+discount*best_q-self.q_func[(old_state,behavior_action)])

                traj.append(behavior_action)
                num_states += 1
                total_reward += reward

            print(f"Episode {episode} with reward {total_reward}")
            
            rewards.append(total_reward)
            lengths.append(num_states)
            
            episode += 1     

        return lengths, rewards


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    agent = TabularQLearnAgent(env,epsilon=0.1)

    agent.init_qfunc_from_states_actions(range(env.observation_space.n),range(env.action_space.n))

    lengths, rewards = agent.train_agent(5000)

    plt.cla()
    plt.plot(lengths)
    plt.savefig("images/ql_lengths.png")
    plt.cla()
    plt.plot(rewards)
    plt.savefig("images/ql_rewards.png")

    #View performance of greedy agent in environment
    agent.set_epsilon(0)
    agent.observe_agent()