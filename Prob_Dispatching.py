import numpy as np
import cvxpy as cp
import pickle


class Dispatching_Parameters:
    def __init__(self, params):
        [users, stations, evs_loc, stations_loc, p_min, p_max, target_evs, require_loads] = params
        self.evs = users
        self.stations = stations
        self.evs_loc = evs_loc
        self.sts_loc = stations_loc
        self.p_min = p_min
        self.p_max = p_max
        self.target_evs = target_evs
        self.evs_load = require_loads


class Leader:
    def __init__(self, initial_price, p_min, p_max, target_evs, stations, stations_loc):
        self.decision = initial_price
        self.p_min = p_min
        self.p_max = p_max
        self.target_evs = target_evs
        self.stations = stations
        self.sts_loc = stations_loc

    def update_grad(self, gradient, step_size):
        self.decision = self.decision + step_size * gradient
        p = cp.Variable(self.stations)
        obj = cp.Minimize(cp.sum(cp.power(p - self.decision, 2)))
        const = [self.p_min*np.ones(self.stations) <= p, p <= self.p_max*np.ones(self.stations)]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.decision = p.value

    def update_direct(self, next_decision):
        self.decision = next_decision
        p = cp.Variable(self.stations)
        obj = cp.Minimize(cp.sum(cp.power(p - self.decision, 2)))
        const = [self.p_min * np.ones(self.stations) <= p, p <= self.p_max * np.ones(self.stations)]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.decision = p.value


class Follower:
    def __init__(self, initial_dest, require_load, distances):
        self.decision = initial_dest
        self.load = require_load
        self.distances = distances


class Dispatching(Dispatching_Parameters):
    step_size = 0.05
    max_iter = 10000
    eps = 1e-6
    ve_step_size = 0.1
    ve_max_iter = 1000
    ve_eps = 1e-6
    active_epsilon = 1e-6

    followers = []
    leader_decision_history = []
    followers_decision_history = []
    leader_utility_history = []
    followers_utility_history = []

    def __init__(self, params, filename=None):
        super().__init__(params)
        self.p_init = (self.p_min + self.p_max)/2 * np.ones(self.stations)
        self.x_init = np.ones((self.evs, self.stations))
        for i in range(self.evs):
            self.followers += [Follower(self.x_init[i], self.evs_load[i], )]

    def compute_followers_ve(self):
        print("VE")
        x_cur = np.zeros((self.evs, self.stations))
        for i in range(self.evs):
            x_cur[i] = self.followers[i].decision