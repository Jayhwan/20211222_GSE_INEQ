import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt
import time

class DispatchingParameters:
    def __init__(self, params):
        [users, stations, evs_loc, stations_loc, p_min, p_max, target_evs, require_loads, priority, energy_limit, ev_limit] = params
        self.evs = users # number of evs
        self.stations = stations # number of stations
        self.evs_loc = evs_loc # location of evs (list of numpy array)
        self.sts_loc = stations_loc # location of stations (list of numpy array)
        self.p_min = p_min # minimum price
        self.p_max = p_max # maximum price
        self.target_evs = target_evs # target number of evs in each station (list of numbers) len sts
        self.evs_load = require_loads # require energy of evs (list of numbers) len evs
        self.distance_matrix = np.zeros((self.evs, self.stations)) # i,j represent distance between ev i and station j
        for i in range(self.evs):
            for j in range(self.stations):
                self.distance_matrix[i, j] = np.sqrt(np.sum(np.power(np.array(self.evs_loc[i]) - np.array(self.sts_loc[j]), 2)))
        self.evs_priority = priority # evs x 3
        self.max_elec = energy_limit # stations
        self.max_evs = ev_limit # stations


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
    def __init__(self, initial_dest, require_load, distances, priority):
        self.decision = initial_dest
        self.load = require_load
        self.distances = distances
        self.priority = priority

    def update(self, next_decision):
        self.decision = next_decision


class History:
    def __init__(self):
        self.leader_decision_history = []
        self.leader_utility_history = []
        self.followers_decision_history = []
        self.followers_utility_history = []
        self.updated_cnt = 0
        self.time_history = []

    def update(self, leader_action, leader_utility, follower_action, follower_utility, time_hist):
        self.leader_decision_history += [leader_action]
        self.leader_utility_history += [leader_utility]
        self.followers_decision_history += [follower_action]
        self.followers_utility_history += [follower_utility]
        self.updated_cnt += 1
        self.time_history += [time_hist]

    def initialize(self):
        self.leader_decision_history = []
        self.leader_utility_history = []
        self.followers_decision_history = []
        self.followers_utility_history = []
        self.updated_cnt = 0
        self.time_history = []


class Dispatching(DispatchingParameters):
    grad_step_size = 1
    grad_max_iter = 100
    grad_eps = 1e-4

    ve_step_size = 0.01
    ve_max_iter = 100000
    ve_eps = 1e-5
    active_epsilon = 1e-6
    prox_gamma = 10
    prox_eps = 1e-6
    prox_max_iter = 1000

    heur_beta = 0.1
    heur_eps = 1e-6
    heur_max_iter = 100

    followers = []
    grad_history = History()
    heur_history = History()
    prox_history = History()

    def __init__(self, params, filename=None):
        super().__init__(params)
        self.p_init = (self.p_min + self.p_max)/2 * np.ones(self.stations)
        self.x_init = np.ones((self.evs, self.stations))/self.stations
        self.leader = Leader(self.p_init, self.p_min, self.p_max, self.target_evs, self.stations, self.sts_loc)
        self.filename = filename
        for i in range(self.evs):
            self.followers += [Follower(self.x_init[i], self.evs_load[i], self.distance_matrix[i], self.evs_priority[i])]

    def draw_map(self):
        plt.figure()
        marker_size = 5
        ev_x = []
        ev_y = []
        st_x = []
        st_y = []
        for ev_loc in self.evs_loc:
            ev_x += [ev_loc[0]]
            ev_y += [ev_loc[1]]
            #plt.plot(ev_loc, marker='o', marker_size=marker_size, color = 'r')
        for st_loc in self.sts_loc:
            st_x += [st_loc[0]]
            st_y += [st_loc[1]]
            #plt.plot(st_loc, marker='o', marker_size=marker_size, color = 'k')
        plt.scatter(ev_x, ev_y, s=marker_size, color='r')
        plt.scatter(st_x, st_y, s=marker_size, color='k')
        plt.show()

    def change_filename(self, filename):
        self.filename = filename
        return 0

    def initialize_action(self):
        self.leader.update_direct(self.p_init)
        self.update_followers(self.x_init)

    def leader_action(self):
        return self.leader.decision

    def followers_action(self):
        action = np.zeros((self.evs, self.stations))
        for i in range(self.evs):
            action[i] = self.followers[i].decision
        return action

    def leader_utility(self):
        dest = self.followers_action()
        obj = np.sum(np.power(self.target_evs - np.sum(dest, axis=0), 2))
        return -obj

    def waiting_time(self):
        return np.sum(self.followers_action(), axis=0)

    def followers_utility(self):
        dest = self.followers_action()
        objs = np.zeros(self.evs)
        for i in range(self.evs):
            objs[i] = self.evs_priority[i, 0]*dest[i]@self.distance_matrix[i].T + self.evs_priority[i, 1]*self.evs_load[i]*dest[i]@self.leader_action().T\
                      + self.evs_priority[i, 0] * dest[i]@self.waiting_time().T
        return objs

    def update_followers(self, x_):
        for i in range(self.evs):
            self.followers[i].update(x_[i])
        return 0

    def save_data(self):
        if self.filename is not None:
            with open(self.filename, "wb") as f:
                pickle.dump(self, f)
            with open(self.filename[:-4]+"_grad_history.pkl", "wb") as f:
                pickle.dump(self.grad_history, f)
            with open(self.filename[:-4]+"_heur_history.pkl", "wb") as f:
                pickle.dump(self.heur_history, f)
            with open(self.filename[:-4]+"_prox_history.pkl", "wb") as f:
                pickle.dump(self.prox_history, f)
            print("SAVED!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 0

    def print_information(self):
        print("Leader Decision :", self.leader_action())
        print("Leader Utility  :", self.leader_utility())
        print("Target EV       :", self.target_evs)
        print("Current EV      :", np.sum(self.followers_action(), axis=0))

        #print("Step size       :", self.grad_step_size)
        #print("EVs Decision    :", self.followers_action())
        return 0

    def compute_followers_ve(self):
        print("Start to compute the VE of the followers")
        x_cur = self.followers_action()
        p = self.leader_action()

        A_coef = np.multiply(np.kron(np.ones((self.stations, 1)), self.evs_priority[:, 0]).T, self.distance_matrix) +\
                 np.multiply(self.evs_priority[:, 1], self.evs_load).reshape(-1, 1)@self.leader_action().reshape(1, -1)
        iter = 0
        for iter in range(self.ve_max_iter):
            Fx = A_coef + np.multiply(np.kron(np.ones(self.stations), self.evs_priority[:, 2].reshape(-1, 1)), np.kron(np.ones((self.evs, 1)), np.sum(x_cur, axis=0)) + x_cur)
            x_next = x_cur - self.ve_step_size * Fx
            #print(x_next)
            x_proj = cp.Variable((self.evs, self.stations))
            obj = cp.Minimize(cp.sum(cp.power(x_next - x_proj, 2)))
            const = []
            const += [np.zeros((self.evs, self.stations)) <= x_proj]
            const += [cp.sum(x_proj, axis=1) == np.ones(self.evs)]
            const += [self.evs_load@x_proj <= self.max_elec]
            const += [cp.sum(x_proj, axis=0) <= self.max_evs]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            x_next = x_proj.value
            if (iter+1)%200 == 0:
                print("ITER", iter+1, "VE GAP :", np.sqrt(np.sum(np.power(x_cur - x_next, 2))))
            if np.sqrt(np.sum(np.power(x_cur - x_next, 2))) <= self.ve_eps and iter >= 200:
                break
            x_cur = x_next
            if (iter+1)%1000 == 0:
                x_check = cp.Variable((self.evs, self.stations))
                obj_check = cp.Minimize(cp.sum(cp.multiply(A_coef + np.multiply(np.kron(np.ones(self.stations), self.evs_priority[:, 2].reshape(-1, 1)), np.kron(np.ones((self.evs, 1)), np.sum(x_next, axis=0)) + x_next), x_check - x_next)))
                const_check = []
                const_check += [np.zeros((self.evs, self.stations)) <= x_check]
                const_check += [cp.sum(x_check, axis=1) == np.ones(self.evs)]
                const_check += [self.evs_load @ x_check <= self.max_elec]
                const_check += [cp.sum(x_check, axis=0) <= self.max_evs]
                prob_check = cp.Problem(obj_check, const_check)
                result = prob_check.solve(solver='ECOS')
                print("ITER :", iter + 1, "VE CHECKING :", result)
        #self.print_information()
        #print("VE Done")
        for i in range(self.evs):
            self.followers[i].update(x_next[i])
        #print("Target  EV :", self.target_evs)
        #print("Current EV :", np.sum(x_next, axis=0))
        return x_next

    def inequality_const_value(self):
        dest = self.followers_action()
        x_nm = - dest
        x_m = np.zeros((2, self.stations))
        x_nm_mu = np.zeros((self.evs, self.stations))
        x_m_mu = np.zeros((2, self.stations))

        x_m[0, :] = np.sum(np.multiply(dest, np.kron(np.ones(self.stations), self.evs_load.reshape(-1, 1))), axis=0) - self.max_elec
        x_m[1, :] = np.sum(dest, axis=0) - self.max_evs

        z_nm = np.copy(x_nm)
        z_m = np.copy(x_m)

        f = lambda x: np.abs(x) <= self.active_epsilon
        x_nm_mu += f(x_nm)
        x_m_mu += f(x_m)
        return [x_nm, x_m, z_nm, z_m, x_nm_mu, x_m_mu]

    def is_active_constraints(self):
        [x_nm, x_m, z_nm, z_m, x_nm_mu, x_m_mu] = self.inequality_const_value()
        f = lambda x: np.abs(x) <= self.active_epsilon
        active = np.hstack((f(x_nm).reshape(-1), f(x_m).reshape(-1), f(z_nm).reshape(-1), f(z_m).reshape(-1), f(x_nm_mu).reshape(-1), f(x_m_mu).reshape(-1)))
        return active

    def compute_leader_gradient(self):
        print("Computing Gradient")
        dest = self.followers_action()
        active = self.is_active_constraints()
        Dxh = np.zeros((2*self.evs*self.stations+2*self.stations+2*self.evs, self.stations))
        for i in range(self.evs):
            Dxh[i*self.stations:(i+1)*self.stations] = self.evs_priority[i, 1]*self.evs_load[i]*np.eye(self.stations)

        Dyh = np.zeros((2*self.evs*self.stations+2*self.stations+2*self.evs, 3*self.evs*self.stations+2*self.stations+self.evs))
        m = self.stations
        n = self.evs
        Dyh[:m*n, :m*n] = np.kron(np.diag(self.evs_priority[:, 2])+np.kron(np.ones((n, 1)), self.evs_priority[:, 2]), np.eye(m))
        Dyh[:m*n, 2*m*n:3*m*n] = -np.eye(m*n)
        Dyh[:m * n, 3 * m * n:3 * m * n + m] = np.kron(self.evs_load.reshape(-1, 1), np.eye(m))
        Dyh[:m * n, 3 * m * n + m:3 * m * n + 2* m] = np.kron(np.ones((n, 1)), np.eye(m))
        Dyh[:m * n, 3 * m * n + 2 * m:3 * m * n + 2 * m + n] = np.kron(np.eye(n), np.ones((m, 1)))

        Dyh[m*n : 2* m * n+2*m, 2 * m * n:3*m*n+2*m] = np.diag(np.hstack((dest.reshape(-1), np.sum(np.multiply(dest, np.kron(np.ones(self.stations), self.evs_load.reshape(-1, 1))), axis=0) - self.max_elec,
                                                                          np.sum(dest, axis=0) - self.max_evs)))
        Dyh[2*m*n+2*m:2*m*n+2*m+n, :m*n] = np.kron(np.eye(n), np.ones(m))

        Dxg = np.zeros((3*m*(n+2), m))
        Dyg = np.zeros((3*m*(n+2), 3*m*n+2*m+n))
        Dyg[:m*n, :m*n] = -np.eye(m*n)
        Dyg[m*n:m*n+2*m, :m*n] = np.kron(np.vstack((self.evs_load, np.ones(n))), np.eye(m))
        #Dyg[m*n+2*m:2*(m*n+2*m), m*n:2*m*n] = Dyg[:m*n+2*m, :m*n]
        Dyg[2*(m*n+2*m):,2*m*n:2*m*n+m*n+2*m] = -np.eye(m*n+2*m)

        Dxh_wave = Dxh
        Dyh_wave = Dyh
        for i in range(len(active)):
            if active[i]:
                Dxh_wave = np.vstack((Dxh_wave, Dxg[i]))
                Dyh_wave = np.vstack((Dyh_wave, Dyg[i]))
        #print(Dyh_wave.shape, Dxh_wave.shape)

        s = time.time()
        eps = 1e-5
        Dy_var = cp.Variable((3*m*n+2*m+n, m))
        obj = cp.Minimize(1)
        const = [Dyh_wave@Dy_var + Dxh_wave <= eps]
        const = [Dyh_wave @ Dy_var + Dxh_wave >= -eps]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        #print(prob.status)
        dy = Dy_var.value
        e = time.time()

        #print("Gradient Computing time :", e-s)
        v = np.sum(dest, axis=0) - self.target_evs

        dxj = np.zeros(m)
        dyj = 2 * np.kron(np.ones(n), v)
        dj = dxj - dyj@dy[:m*n, :]
        #print(dy)

        print("DJ :", dj)
        return dj

    def grad_one_iteration(self):
        s = time.time()
        x = self.compute_followers_ve()
        e = time.time()
        a = e-s
        print("VE TIME :", e-s)
        self.update_followers(x)
        #print(len(self.grad_history.leader_decision_history))
        #self.update_grad_history(self.leader_action(), self.leader_utility(), self.followers_action(),
        #                         self.followers_utility())
        #print(len(self.grad_history.leader_decision_history))
        p_prev = self.leader_action()
        s2 = time.time()
        grad = self.compute_leader_gradient()
        e2 = time.time()
        b = e2-s2
        print("Grad TIME :", e2-s2)
        if len(self.grad_history.time_history) == 0:
            self.grad_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), a+b)
        else:
            self.grad_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), a + b - self.grad_history.time_history[-1])
        self.save_data()
        self.leader.update_grad(grad, self.grad_step_size)
        diff1 = np.sqrt(np.sum(np.power(self.leader_action() - p_prev, 2)))
        diff2 = np.sqrt(np.sum(np.power(grad, 2)))
        print("Sum Time :", a+b)
        return diff1, diff2

    def grad_iterations(self):
        if self.grad_history.updated_cnt:
            self.leader.update_direct(self.grad_history.leader_decision_history[-1])
            for i in range(self.evs):
                self.update_followers(self.grad_history.followers_decision_history[-1])
        else:
            self.initialize_action()

        for i in range(self.grad_max_iter):
            print("Already", self.grad_history.updated_cnt, "iteration progressed")
            print("Grad Iteration :", i+1)
            self.print_information()
            diff1, diff2 = self.grad_one_iteration()

            if (np.abs(diff1 < self.grad_eps) or np.abs(diff2 < self.grad_eps))and i > 10:
                #self.save_data()
                print("Grad Iteration Over")
                print("grad diff :", np.abs(diff1), "action diff :", np.abs(diff2), "smaller than eps :", self.grad_eps)
                break
            #if i % 1 == 0:
            #    print("diff :", diff)
            #    self.save_data()
        print("Grad Maximum Iteration Over")
        self.save_data()
        return 0

    def heur_one_iteration(self):
        p_prev = self.leader_action()
        x = self.compute_followers_ve()
        print(np.sum(x, axis=0))
        self.update_followers(x)
        target_diff = np.sum(x, axis=0) - self.target_evs
        update = self.heur_beta * np.multiply(np.sign(target_diff), np.power(target_diff, 2))
        p_next = p_prev + update
        p_next = np.maximum(p_next, self.p_min * np.ones(self.stations))
        p_next = np.minimum(p_next, self.p_max * np.ones(self.stations))
        self.heur_beta *= 0.95
        return p_next

    def heur_iterations(self):
        if self.heur_history.updated_cnt:
            self.leader.update_direct(self.heur_history.leader_decision_history[-1])
            self.update_followers(self.heur_history.followers_decision_history[-1])
        #else:
            #self.initialize_action()
        for i in range(self.heur_max_iter):
            print("Already", self.heur_history.updated_cnt, "iteration progressed")
            print("Heur Iteration :", i+1)

            self.print_information()

            next_p = self.heur_one_iteration()
            diff = np.sqrt(np.sum(np.power(self.leader_action() - next_p, 2)))

            self.heur_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), 0)
            self.save_data()
            self.leader.update_direct(next_p)

            if np.abs(diff) < self.heur_eps:
                self.save_data()
                print("Heur Iteration Over")
                print("diff :", np.abs(diff), "smaller than eps :", self.heur_eps)
                break
            if i % 1 == 0:
                print("diff :", diff)
                #self.save_data()
        print("Heur Maximum Iteration over")
        self.save_data()
        return 0

    def prox_one_iteration(self):
        x = self.followers_action()
        #self.update_prox_history(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility())
        x_prev = np.copy(x)
        p_prev = self.leader_action() # leader action doesn't change
        """
        p_var = cp.Variable(self.stations)
        obj = cp.Minimize(cp.sum(cp.power(self.target_evs - np.sum(x_prev, axis=0), 2)) + self.prox_gamma/2*cp.sum(cp.power(p_prev - p_var, 2)))

        const = [self.p_min*np.ones(self.stations) <= p_var, p_var <= self.p_max*np.onse(self.stations)]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.leader.update_direct(p_var.value)
        """

        for i in range(self.evs):
            x_var = cp.Variable(self.stations)
            pri = self.evs_priority[i]
            help_ = np.ones(self.evs)
            help_[i] = 0
            obj = cp.Minimize(pri[2]*cp.sum(cp.power(x_var, 2))+cp.sum(cp.multiply(pri[0]*self.distance_matrix[i]+pri[1]*self.evs_load[i]*self.leader_action()+pri[2]*help_@x_prev, x_var))
                              + self.prox_gamma/2*cp.sum(cp.power(x_prev[i] - x_var, 2)))
            const = []
            const += [np.zeros(self.stations)<=x_var, cp.sum(x_var) == 1]
            help_1 = np.copy(self.evs_load)
            help_1[i] = 0
            const += [help_1@x_prev + self.evs_load[i]*x_var <= self.max_elec]
            const += [help_@x_prev + x_var <= self.max_evs]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            #print(x_var.value)
            x_prev[i] = x_var.value
            #print(x_prev[i], x_var.value)
            self.followers[i].update(x_var.value)
        diff = np.sqrt(np.sum(np.power(self.leader.decision - p_prev, 2) + np.power(x_prev - x, 2)))
        return diff

    def prox_iterations(self):
        if self.prox_history.updated_cnt:
            self.leader.update_direct(self.prox_history.leader_decision_history[-1])
            self.update_followers(self.prox_history.followers_decision_history[-1])
        else:
            self.initialize_action()
            self.prox_history.update(self.leader_action(), self.leader_utility(), self.followers_action(),
                                     self.followers_utility(), 0)
        s = time.time()
        for i in range(self.prox_max_iter):
            print("Already", self.prox_history.updated_cnt, "iteration progressed")
            print("Prox Iteration :", i+1)
            self.print_information()
            diff = self.prox_one_iteration()
            c = time.time()
            self.prox_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), c-s)
            self.save_data()
            if i % 1 == 0:
                print("diff :", diff)
            #    self.save_data()
            if np.abs(diff) < self.prox_eps:
                print("Prox Iteration Over")
                print("diff :", np.abs(diff), "smaller than eps :", self.prox_eps)
                return 0
        print("Prox Maximum Iteration over")
        self.save_data()
        return 0
"""
ev_ = 5
sts_ = 5
ev_loc = []
station_loc = []
for i in range(1, ev_+1):
    for j in range(1, ev_+1):
        ev_loc += [np.array([i-0.5*np.random.random(), j-0.5*np.random.random()])]
for a, b in [[0.2, 1], [1, 4], [3, 2], [4, 4], [4.5, 0.3]]:
    station_loc += [np.array([a+0.5*np.random.random(), b+0.5*np.random.random()])]
#for i in range(25):
#    ev_loc += [np.array([5*np.random.random(), 5*np.random.random()])]
target_ev = [3, 6, 8, 5, 3]
load = 0.8*np.random.random(ev_**2)+0.2
elec_max = 3*np.random.random(sts_)+4
ev_max = 12*np.ones(sts_)
#print(len(ev_loc), len(station_loc), len(elec_max), len(ev_max))
parameter = [25, 5, ev_loc, station_loc, 1, 20, target_ev, load, np.ones((ev_**2, 3))/3, elec_max, ev_max]
prob = Dispatching(parameter, "test_dispatching.pkl")
prob.draw_map()
prob.grad_iterations()
"""
"""
#print("Leader Action")
#print(prob.leader_action())
print(prob.leader_utility())
#print("Follower Action")
#print(prob.followers_action())
#print(prob.followers_utility())
print("Require load :", np.sum(load), "Possible load :", np.sum(elec_max))
print("Require space :", ev_**2, "Enough space :", np.sum(ev_max))

print(np.sum(prob.followers_action(), axis=0))
prob.draw_map()
#print(prob.waiting_time())
print(prob.compute_followers_ve())
print(np.sum(prob.followers_action(), axis=0))
print(prob.leader_utility())
prob.leader.update_direct([20, 6, 0, 10, 10])
print(prob.leader_utility())
print(prob.compute_followers_ve())
print(prob.leader_utility())
print(np.sum(prob.followers_action(), axis=0))
"""