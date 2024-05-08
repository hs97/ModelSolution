import numpy as np
from gym.spaces import Box


class GrowthModel:
    def __init__(self,
                δ=0.1,        # capital depreciation
                β=0.96,       # discount factor
                μ=0,          # shock location parameter
                k_max=4,      # max capital
                s=0.1,        # shock scale parameter
                c_high=1,     # consumption bound
                α=0.4       # capital share of output
                ):

        self.β, self.μ, self.s, self.k_max, self.δ = β, μ, s, k_max, δ
        self.α = α
        # Use gym action space to accomodate continuous actions
        self.action_space = Box(low=0, high=c_high, shape=(1, ), dtype=np.float32)


    def f(self, k):
        return np.nan_to_num(np.power(k, α), nan=-np.inf)
    
    def u(self, c):
        return np.log(c)

    def reset(self):
        """
        Resets the current_state variable.
        """
        k_init = np.random.beta(5, 5)
        shock = self.μ + self.s*np.random.randn()
        self.current_state = np.array([k_init, shock])
        self.count = 0
        return self.current_state

    def step(self, action: float):
        """
        Takes a step in the environment by sampling from the
        transition matrix self.T given an action and the current_state and return the index of the next state.
        It returns a tuple of (next_state, reward, terminal, truncation, flags).
        """
        assert self.current_state is not None, "State has not been reset"
        terminal, truncation = False, False
        # Current period states
        k, shock = self.current_state
        # Compute macro variables
        y = np.exp(shock)*self.f(k)
        c = action[0]*y
        reward = self.u(c)
        # Next period states
        next_k = y - c + (1 - self.δ)*k
        shock = self.μ + self.s*np.random.randn()
        self.current_state = np.array([next_k, shock])
        self.count += 1
        if self.count >= 1000:
            truncation = True
        if next_k == 0:
            terminal = True
        return self.current_state, reward, terminal, truncation

    @property
    def features_size(self):
        return self.phi(0).shape[0]


class RBC:
    def __init__(self,
                δ=0.1,        # capital depreciation
                β=0.96,       # discount factor
                μ=0,          # shock location parameter
                k_max=4,      # max capital
                s=0.1,        # shock scale parameter
                c_high=1,     # consumption bound
                α=0.4
                ):

        self.u, self.f, self.β, self.μ, self.s, self.k_max, self.δ = β, μ, s, k_max, δ
        # Use gym action space to accomodate continuous actions
        self.α = α
        self.action_space = Box(low=0, high=c_high, shape=(2, ), dtype=np.float32)
        # What if, isntead of using percentage, we use number (bounding max consumption by 3)
        # self.action_space = Box(low=0, high=3, shape=(1, ), dtype=np.float32)

    def f(k, n):
        return np.nan_to_num(np.power(k, self.α) * np.power(n, 1-self.α), nan=-np.inf)

    def u(self, c, n):
        return np.log(c) - np.log(n)  

    def fcd_n(self, k, n):
        return (1-self.α)*np.nan_to_num(np.power(k, α) * np.power(n, -α), nan=-np.inf)
    def reset(self):
        """
        Resets the current_state variable.
        """
        k_init = np.random.beta(5, 5)
        shock = self.μ + self.s*np.random.randn()
        self.current_state = np.array([k_init, shock])
        self.count = 0
        return self.current_state

    def step(self, action: float):
        """
        Takes a step in the environment by sampling from the
        transition matrix self.T given an action and the current_state and return the index of the next state.
        It returns a tuple of (next_state, reward, terminal, truncation, flags).
        """
        assert self.current_state is not None, "State has not been reset"
        terminal, truncation = False, False
        # Current period states
        k, shock = self.current_state
        # Compute macro variables
        n = action[1]
        y = np.exp(shock)*self.f(k, n)
        c = action[0]*y
        reward = self.u(c, n)
        # Next period states
        next_k = y - c + (1 - self.δ)*k
        shock = self.μ + self.s*np.random.randn()
        self.current_state = np.array([next_k, shock])
        self.count += 1
        if self.count >= 1000:
            truncation = True
        if next_k == 0:
            terminal = True
        return self.current_state, reward, terminal, truncation

    @property
    def features_size(self):
        return self.phi(0).shape[0]