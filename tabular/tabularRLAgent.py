import time

class TabularRLAgent(object):
    def __init__(self,env, epsilon=0.2):
        self.q_func = None
        self.epsilon = epsilon
        self.env = env
    
    def set_qfunc(self,q_func):
        if type(q_func) == type({}):
            self.q_func = q_func
        else:
            raise Exception(f"Q function should be a dict, not a {type(q_func)}")

    def init_qfunc_from_states_actions(self,states,actions, init_val=0):
        """
        Initializes Agent's Q function to a specified value at all
        state action pairs.
        """
        self.q_func = {}
        for action in actions:
            for state in states:
                state_action_pair = (state, action)
                self.q_func[state_action_pair] = init_val
    
    def set_epsilon(self,epsilon):
        self.epsilon = epsilon

    def observe_agent(self):
        terminated = False
        observation = self.env.reset()
        self.env.render('human')
        while not terminated:
            actions = range(self.env.action_space.n)
            action = self.act_behavior_policy(actions,observation)

            observation, reward, terminated, info = self.env.step(action)
            self.env.render('human')
            time.sleep(1)