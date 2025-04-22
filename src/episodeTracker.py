class EpisodeTracker:
    def __init__(self, allRecord=False):
        self.reset(allRecord)

    def reset(self,allRecord):
        """Resets all tracking variables for a new episode"""
        self.latest_observation = None
        self.past_observations = []  # Stores last 5 observations (FIFO)
        self.all_observations = []   # Stores all observations for debug
        self.actions = []
        self.rewards = []
        self.done = False
        self.allRecord = allRecord
        self.epStep = 0

    def add_observation(self, observation):
        """Updates observation history while maintaining 5-step window"""
        if self.latest_observation is not None:
            self.past_observations.append(self.latest_observation)
            # Maintain only last 5 observations
            if len(self.past_observations) > 5:
                self.past_observations.pop(0)
        self.latest_observation = observation
        if self.allRecord:
            self.all_observation(observation)
        self.epStep =+ 1 # increment step counter

    def all_observation(self, observation):
        """Updates observation history for all observations"""
        self.all_observations.append(observation)

    def add_action(self, action):
        """Records action taken by the agent"""
        self.actions.append(action)

    def add_reward(self, reward):
        """Records reward received from environment"""
        self.rewards.append(reward)

    def get_latest_observation(self):
        return self.latest_observation

    def get_past_observations(self):
        return self.past_observations.copy()

    def get_actions(self):
        return self.actions.copy()

    def get_rewards(self):
        return self.rewards.copy()