import numpy as np


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed in the current state. Note that this is a different formula
        for the step size than was used in MultiArmedBandits. Use an
        epsilon-greedy policy for action selection. Note that unlike the
        pseudocode, we are looping over a total number of steps, and not a
        total number of episodes. This allows us to ensure that all of our
        trials have the same number of steps--and thus roughly the same amount
        of computation time.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - Use the provided self._get_epsilon function whenever you need to
            obtain the current value of epsilon.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        state_action_values = np.zeros((env.observation_space.n, env.action_space.n))
        Nsa = np.ones((env.observation_space.n, env.action_space.n))
        action = env.action_space.sample()
        observation = env.reset()
        rewards = np.zeros(100)
        rewards_raw = np.zeros((100,int(steps/100)))
        for s in range(steps):
            #env.render()
            if np.random.rand() < self._get_epsilon(s/steps):
                action = env.action_space.sample() #[np.random.randint(0,env.action_space.n)]
            else:
                #action = np.argmax(state_action_values[observation])
                m = max(state_action_values[observation])
                action = np.random.choice([i for i, j in enumerate(state_action_values[observation]) if j == m]) #choose randomly among all of the actions that are tied for best

            Nsa[observation,action] += 1 #increment N(s,a)

            new_observation, reward, done, info = env.step(action) #perform action, get feedback
            #print(reward)
            Q=state_action_values[observation, action]
            Q_next_max = np.max(state_action_values[new_observation]) #state_action_values[new_observation, np.argmax(state_action_values[new_observation])]
            N=Nsa[observation,action]
            state_action_values[observation, action] = Q+1/N*(reward + self.discount*Q_next_max - Q) #update state value

            observation=new_observation

            r = int(np.floor(steps / 100))
            i = int(np.floor(s / r)) #number from one to 100
            rewards_raw[i, s%r] = reward
            #rewards[i] = old average*((s)%r)/((s+1)%r) + reward/((s+1)%r) #(rewards[i]*((s-1)%100)+reward)/s%100 #rewards[i]+1/(s+1)*(reward-rewards[i])


            if done:
                observation = env.reset()

        env.close()
        rewards = np.mean(rewards_raw, axis=1)
        return state_action_values, rewards


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        observation = env.reset()
        actions = []
        states = []
        rewards = []
        for s in range(10000): #arbitrarily large number to prevent infinite loop
            m = max(state_action_values[observation])
            action = np.random.choice([i for i, j in enumerate(state_action_values[observation]) if j == m]) #choose randomly among all of the actions that are tied for best
            #action = np.argmax(state_action_values[observation])
            actions.append(action)
            observation, reward, done, info = env.step(action) #perform action, get feedback
            rewards.append(reward)
            states.append(observation)
            if done:
                break

        env.close()

        return np.array(states), np.array(actions), np.array(rewards)


    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return (1 - progress) * self.epsilon
