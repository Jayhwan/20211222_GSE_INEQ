import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt


class Dispatching_Parameters:
    def __init__(self, params):
        [users, stations, evs_loc, stations_loc, p_min, p_max, target_evs, require_loads, weight] = params
        self.evs = users # number of evs
        self.stations = stations # number of stations
        self.evs_loc = evs_loc # location of evs (list of numpy array)
        self.sts_loc = stations_loc # location of stations (list of numpy array)
        self.p_min = p_min # minimum price
        self.p_max = p_max # maximum price
        self.target_evs = target_evs # target number of evs in each station (list of numbers) len sts
        self.evs_load = require_loads # require energy of evs (list of numbers) len evs
        self.weight = weight #
        self.distance_matrix = np.zeros((self.evs, self.stations)) # i,j represent distance between ev i and station j
        for i in range(self.evs):
            for j in range(self.stations):
                self.distance_matrix[i, j] = np.sqrt(np.power(self.evs_loc[i] - self.sts_loc[j], 2))


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

    def update(self, next_decision):
        self.decision = next_decision


class History:
    def __init__(self):
        self.leader_decision_history = []
        self.leader_utility_history = []
        self.followers_decision_history = []
        self.followers_utility_history = []


class Dispatching(Dispatching_Parameters):
    step_size = 0.05
    max_iter = 10000
    eps = 1e-6
    ve_step_size = 0.1
    ve_max_iter = 1000
    ve_eps = 1e-6
    active_epsilon = 1e-6

    followers = []
    grad_history = History()
    baseline_1_history = History()
    baseline_2_history = History()

    def __init__(self, params, filename=None):
        super().__init__(params)
        self.p_init = (self.p_min + self.p_max)/2 * np.ones(self.stations)
        self.x_init = np.ones((self.evs, self.stations))/self.stations
        self.leader = Leader(self.p_init, self.p_min, self.p_max, self.target_evs, self.stations, self.sts_loc)
        self.filename = filename
        for i in range(self.evs):
            self.followers += [Follower(self.x_init[i], self.distance_matrix[i], self.evs_load[i])]

    def draw_map(self):
        plt.figure()
        marker_size = 5
        for ev_loc in self.evs_loc:
            plt.plot(ev_loc, marker='o', marker_size=marker_size, color = 'r')
        for st_loc in self.sts_loc:
            plt.plot(st_loc, marker='o', marker_size=marker_size, color = 'k')
        plt.show()

    def followers_action(self):
        action = np.zeros((self.evs, self.stations))
        for i in range(self.evs):
            action[i] = self.followers[i].decision
        return action

    def leader_utility(self):
        dest = self.followers_action()
        obj = np.sum(np.power(self.target_evs - np.sum(dest, axis=0)))
        return -obj

    def waiting_time(self):
        t


    def followers_utility(self):
        [a,b,c] = self.weight
        dest = self.followers_action()
        objs = np.zeros(self.evs)
        for i in range(self.evs):
            objs[i] = a*dest[i]@self.distance_matrix[i].T + b*self.evs_load[i]*dest[i]@self.leader.decision.T\
                      + c*

