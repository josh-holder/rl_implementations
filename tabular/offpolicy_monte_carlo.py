import gym
import numpy as np
import random
from matplotlib import pyplot as plt
import time
from tabularRLAgent import TabularRLAgent

class TabularOnPolicyMonteCarloAgent(TabularRLAgent):
    def __init__(self,env,epsilon=0.2):
        super().__init__(env,epsilon=epsilon)

        #Initialize storage structures for # visits for each state action pair
        self.cumulative_weights = {}
        self.num_visits_to_state_action = {}
        for action in range(self.env.action_space.n):
            for state in range(self.env.observation_space.n):
                self.num_visits_to_state_action[(state,action)] = 0
                self.cumulative_weights[(state,action)] = 0
        

    def act_behavior_policy(self, actions, state):
        """
        Behavior policy is e-greedy w/r/t q function.
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

    def act_target_policy(self,actions,state):
        #for on-policy monte-carlo, the target policy is also e-greedy
        if not self.off_policy: 
            return self.act_behavior_policy(actions, state)
        else:
            best_q_val = -np.inf
            best_act = None
            for action in actions:
                state_action_pair = (state,action)

                if self.q_func[state_action_pair] > best_q_val:
                    best_q_val = self.q_func[state_action_pair]
                    best_act = action

            return best_act

    def train_agent_onpolicy(self, episodes, discount=1):
        """
        Trains the monte carlo agent on-policy for the specified amount of episodes, with the specified
        hyperparameters.

        Returns lists of rewards and lengths for which to use in tracking progress of agents.
        """
        episode = 0

        lengths = []
        total_rewards = []

        avg_rewards = 0
        avg_length = 0
        while episode < episodes:
            #generate full trajectory according to behavior policy
            state = self.env.reset()
            rewards = []
            states = []
            behavior_actions = []
            target_actions = []
            terminated = False
            while not terminated:
                action_options = range(self.env.action_space.n)
                behavior_action = self.act_behavior_policy(action_options,state)
                target_action = self.act_target_policy(action_options,state)
                state, reward, terminated, info = self.env.step(behavior_action)

                rewards.append(reward)
                states.append(state)
                behavior_actions.append(behavior_action)
                target_actions.append(target_action)
            
            avg_rewards += sum(rewards)
            avg_length += len(states)
            if episode % 10 == 0:
                print(f"Episode {episode} with reward {avg_rewards/10}, length {avg_length/10}")
                print(self.q_func[(25,0)],self.q_func[(25,1)],self.q_func[(25,2)])
                avg_rewards = 0
                avg_length = 0
                

            reward = 0
            weight = 1
            #iterate backward through trajectory
            for i in range(len(rewards)-1,-1,-1):
                reward = reward*discount+rewards[i]
                state_action_pair = (states[i],behavior_actions[i])
                self.cumulative_weights[state_action_pair] += weight

                self.num_visits_to_state_action[state_action_pair] += 1
                self.q_func[state_action_pair] += weight/self.cumulative_weights[state_action_pair]*(reward-self.q_func[state_action_pair])

                if behavior_actions[i-1] != target_actions[i-1]:
                    break
                    
                #If the behavior is equal to the target policy, by definition it is the target policy.
                #Thus, it occurs with probability (1-e)+e/num_actions
                weight = weight/((1-self.epsilon)+self.epsilon/self.env.action_space.n)

            lengths.append(len(states))
            total_rewards.append(sum(rewards))
            episode += 1     

        return lengths, total_rewards

    def train_agent_offpolicy(self, episodes, discount=1):
        """
        Trains the monte carlo agent off-oplicy for the specified amount of episodes, with the specified
        hyperparameters.

        Returns lists of rewards and lengths for which to use in tracking progress of agents.
        """
        episode = 0

        # init_epsilon = self.epsilon

        lengths = []
        total_rewards = []

        avg_rewards = 0
        avg_length = 0
        while episode < episodes:
            # self.epsilon -= init_epsilon/episodes

            #generate full trajectory according to behavior policy
            state = self.env.reset()
            rewards = []
            states = []
            actions = []
            terminated = False
            while not terminated:
                action_options = range(self.env.action_space.n)
                behavior_action = self.act_behavior_policy(action_options,state)
                state, reward, terminated, info = self.env.step(behavior_action)

                rewards.append(reward)
                states.append(state)
                actions.append(behavior_action)
            
            avg_rewards += sum(rewards)
            avg_length += len(states)
            if episode % 10 == 0:
                print(f"Episode {episode} with reward {avg_rewards}, length {avg_length}")
                print(self.q_func[(25,0)],self.q_func[(25,1)],self.q_func[(25,2)])
                avg_rewards = 0
                avg_length = 0
                

            reward = 0
            visited_state_action_pairs = []
            #iterate backward through trajectory
            for i in range(len(rewards)-1,-1,-1):
                reward = reward*discount+rewards[i]

                state_action_pair = (states[i],actions[i])

                if state_action_pair not in visited_state_action_pairs:
                    visited_state_action_pairs.append(state_action_pair)
                    self.num_visits_to_state_action[state_action_pair] += 1
                    self.q_func[state_action_pair] += (reward-self.q_func[state_action_pair])/self.num_visits_to_state_action[state_action_pair]
                else:
                    continue

            lengths.append(len(states))
            total_rewards.append(sum(rewards))
            episode += 1     

        return lengths, total_rewards

    

if __name__ == "__main__":
    env = gym.make("CliffWalking-v0")
    agent = TabularMonteCarloAgent(env,off_policy=False)
    
    agent.init_qfunc_from_states_actions(range(env.observation_space.n),range(env.action_space.n))
    
    lengths, rewards = agent.train_agent_onpolicy(5000)

    plt.cla()
    plt.plot(lengths)
    plt.savefig("images/ql_lengths.png")
    plt.cla()
    plt.plot(rewards)
    plt.savefig("images/ql_rewards.png")

    #View performance of agent in environment
    agent.observe_agent()
    