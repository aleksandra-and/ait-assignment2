import random

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1




class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = {}

        for s in range(self.num_states):
            for a in range(self.num_actions):
                if (s not in self.Q):
                    self.Q[s] = {}
                if (a not in self.Q[s]):
                    self.Q[s][a] = 0



    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if (s not in self.Q):
                    self.Q[s] = {}
                if (a not in self.Q[s]):
                    self.Q[s][a] = 0



    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        Q_max = 0

        for a in self.Q[next_state]:
            Q_max = max(Q_max, self.Q[next_state][a])
        if state == next_state:
            reward = -5
        self.Q[state][action] = (1 - self.learning_rate)*self.Q[state][action] + \
                                self.learning_rate * (reward + self.discount*Q_max)
        if done:
            self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + self.learning_rate*reward
        return done


    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """
        prob = random.random()

        if prob < EPSILON:
            return random.randint(0, self.num_actions-1)
        else:
            max_v = -1e8
            for a in self.Q[state]:
                if max_v < self.Q[state][a]:
                    max_v = self.Q[state][a]
            actions = []
            for a in self.Q[state]:
                if max_v == self.Q[state][a]:
                    actions.append(a)
            action_num = random.randint(0, len(actions)-1)
            return actions[action_num]




    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        for s in range(self.num_states):
            print("-------- \n state = ", s)
            for a in range(self.num_actions):
                print("a: ", a, " Q = ", self.Q[s][a], end = ' | ')
            print()










        
