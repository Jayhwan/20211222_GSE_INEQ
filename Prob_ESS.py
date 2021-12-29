import numpy as np
import cvxpy as cp
import pickle


class EET_Parameters:
    def __init__(self, params):
        [users, times, alpha, beta, base_load, p_min, p_max, p_avg, x_mins, x_maxs, require_loads] = params
        self.ev_num = users
        self.time = times
        self.grid_alpha = alpha
        self.grid_beta = beta
        self.grid_load = base_load
        self.p_min = p_min
        self.p_max = p_max
        self.p_avg = p_avg
        self.evs_min = x_mins
        self.evs_max = x_maxs
        self.evs_load = require_loads


class EET_Game(EET_Parameters):
    step_size = 0.1 # Gradient descent step size
    max_iter = 10000
    eps = 1e-10
    ve_step_size = 0.1
    ve_max_iter = 1000
    ve_eps = 1e-6
    active_epsilon = 1e-6
    prox_gamma = 10
    prox_eps = 1e-6
    prox_max_iter = 10000

    followers = []
    leader_decision_history = []
    followers_decision_history = []
    leader_utility_history = []

    def __init__(self, params, filename=None):
        super().__init__(params)
        self.p_init = self.p_avg*np.ones(self.time)  # Initial price of the leader
        self.x_init = np.multiply(np.divide(self.evs_max.T, np.sum(self.evs_max, axis=1)), self.evs_load).T
        self.leader = Leader(self.p_init, self.p_min, self.p_max, self.p_avg)
        self.filename = filename
        for i in range(self.ev_num):
            self.followers += [Follower(self.x_init[i], self.evs_min[i], self.evs_max[i], self.evs_load[i])]

    def compute_followers_ve(self): # and return
        print("VE")
        x_cur = np.zeros((self.ev_num, self.time))
        for i in range(self.ev_num):
            x_cur[i] = self.followers[i].decision
        p = self.leader.decision
        x_next = x_cur
        for iter in range(self.ve_max_iter):
            #print("x_cur :", np.sum(x_cur, axis=0))
            Fx = self.grid_beta*x_cur+np.kron(np.ones((self.ev_num, 1)), self.grid_alpha*np.ones(self.time)+self.grid_beta*(self.grid_load+np.sum(x_cur, axis=0))+p)
            # print("Fx :", Fx)
            x_next = x_cur - self.ve_step_size * Fx

            x_proj = cp.Variable((self.ev_num, self.time))
            obj = cp.Minimize(cp.sum(cp.power(x_next - x_proj, 2)))
            const = []
            const += [self.evs_min <= x_proj, x_proj <= self.evs_max, cp.sum(x_proj, axis=1) == self.evs_load]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            x_next = x_proj.value
            for i in range(self.ev_num):
                self.followers[i].update(x_next[i])
            if iter%200 == 0:
                print("ITER", iter+1, "VE GAP :", np.sqrt(np.sum(np.power(x_cur - x_next, 2))))
            if np.sqrt(np.sum(np.power(x_cur - x_next, 2))) <= self.ve_eps and iter >= 1000-1 :
                break
            x_cur = x_next
        self.print_information()
        print("VE Done")
        return x_next

    def print_information(self):
        print("Leader Decision :", self.leader.decision)
        print("Leader Utility  :", self.leader_utility())
        print()
        x = []
        for f in self.followers:
            x += [f.decision]
        for i in range(self.ev_num):
            continue#print("EVs Decison     :", x[i])

    def inequality_const_value(self):
        v = np.zeros((4, self.time, self.ev_num))
        for i in range(self.ev_num):
            v[0, :, i] = - self.followers[i].decision + self.evs_min[i]
            v[1, :, i] = self.followers[i].decision - self.evs_max[i]
            v[2, :, i] = np.ones(self.time) - v[0, :, i]
            v[3, :, i] = np.ones(self.time) - v[1, :, i]
        return v

    def is_active_inequality_const(self):
        v = self.inequality_const_value()
        f = lambda x: np.abs(x) <= self.active_epsilon
        active = f(v).reshape(-1)
        return active

    def compute_leader_gradient(self): # and return
        print("Gradient")
        active = self.is_active_inequality_const()
        Dxh = np.kron(np.vstack((np.eye(self.time), np.zeros(self.time))), np.ones((self.ev_num, 1)))
        Dxg = np.zeros((np.sum(active), self.time))
        Dxh_wave = np.vstack((Dxh, Dxg))

        Dyh = np.zeros((self.ev_num*(self.time+1), self.ev_num*(3*self.time+1)))
        Dyh[:self.ev_num*self.time, :self.ev_num*self.time] = self.grid_beta*np.kron(np.eye(self.time), np.ones((self.ev_num, self.ev_num))+np.eye(self.ev_num))
        Dyh[self.ev_num*self.time:, :self.ev_num*self.time] = np.kron(np.ones(self.time), np.eye(self.ev_num))
        Dyh[:self.ev_num*self.time, self.ev_num*self.time:3*self.ev_num*self.time] = np.concatenate((-np.eye(self.ev_num*self.time), np.eye(self.ev_num*self.time)), axis=1)
        Dyh[:self.ev_num*self.time, 3*self.ev_num*self.time:] = np.kron(np.ones((self.time, 1)), np.eye(self.ev_num))

        Dyg = np.zeros((4*self.ev_num*self.time, self.ev_num*(3*self.time+1)))
        Dyg[:, :3*self.ev_num*self.time] = np.kron(np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,0,1]]), np.eye(self.ev_num*self.time))

        Dyh_wave = Dyh
        #print(active)
        for i in range(len(active)):
            if active[i]:
                Dyh_wave = np.vstack((Dyh_wave, Dyg[i]))

        Dy_var = cp.Variable((self.ev_num*(3*self.time+1), self.time))
        #print("SHAPE :", Dxh_wave.shape, Dyh_wave.shape, Dy_var.shape)
        obj = cp.Minimize(1)
        const = [Dyh_wave@Dy_var + Dxh_wave == 0]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        dy = Dy_var.value

        dxj = np.zeros(self.time)
        for ev in self.followers:
            dxj += ev.decision

        dyj = np.zeros(self.ev_num*(3*self.time+1))
        dyj[:self.ev_num*self.time] = np.kron(self.leader.decision, np.ones(self.ev_num))
        dj = dxj + dyj@dy
        print("DJ :", dj)
        return dj

    def one_iteration(self): # 현재의 leader action에 대해서 follower들의 ve를 구한 후 gradient를 구해서 leader update
        self.leader_decision_history += [self.leader.decision]
        ve = self.compute_followers_ve()
        self.followers_decision_history += [ve]
        self.leader_utility_history += [self.leader_utility()]
        for i in range(self.ev_num):
            self.followers[i].update(ve[i])
        grad = self.compute_leader_gradient()
        self.leader.update(grad, self.step_size)
        next_leader_decision = self.leader.decision
        diff = np.sqrt(np.sum(np.power(next_leader_decision - self.leader_decision_history[-1], 2)))
        return diff

    def iterations(self):
        for i in range(self.max_iter):
            print("ITERATION :", i+1)
            diff = self.one_iteration()
            self.save_data()
            if np.abs(diff) < self.eps:
                break
        return 0

    def leader_utility(self):
        s = 0
        for t in range(self.time):
            st = 0
            for i in range(self.ev_num):
                st += self.followers[i].decision[t]
            s += self.leader.decision[t]*st
        return s

    def followers_utility(self):
        x = np.zeros((self.ev_num, self.time))
        for i in range(self.ev_num):
            x[i] = self.followers[i].decision

        u = np.zeros(self.ev_num)
        for i in range(self.ev_num):
            u[i] = np.sum(np.multiply(self.grid_alpha*np.ones(self.time)+self.grid_beta*(self.grid_load+np.sum(x, axis=0))+self.leader.decision, x[i]))
        return u

    def save_data(self):
        if self.filename is not None:
            with open(self.filename, "wb") as f:
                pickle.dump(self, f)
        return 0

    def prox_one_iteration(self):
        self.leader_decision_history += [self.leader.decision]
        x = np.zeros((self.ev_num, self.time))
        for i in range(self.ev_num):
            x[i] = self.followers[i].decision

        self.followers_decision_history += [x]
        self.leader_utility_history += [self.leader_utility()]
        x_prev = x
        p_prev = self.leader.decision
        p_var = cp.Variable(self.time)
        obj = cp.Maximize(cp.sum(cp.multiply(p_var, np.sum(x, axis=0))) - self.prox_gamma/2*cp.sum(cp.power(p_prev - p_var, 2)))
        const = [self.p_min*np.ones(self.time) <= p_var, p_var <= self.p_max*np.ones(self.time), cp.sum(p_var) == self.time*self.p_avg]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.leader.decision = p_var.value
        for i in range(self.ev_num):
            x_var = cp.Variable(self.time)
            coef = self.grid_alpha*np.ones(self.time)+self.grid_beta*self.grid_load+self.leader.decision+self.grid_beta*(np.sum(x, axis=0)-x[i])
            obj = cp.Minimize(cp.sum(self.grid_beta*cp.power(x_var, 2)+cp.multiply(coef, x_var)) + self.prox_gamma/2*cp.sum(cp.power(x_var - x[i], 2)))
            const = [self.evs_min[i] <= x_var, x_var <= self.evs_max[i], cp.sum(x_var) == self.evs_load[i]]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            x[i] = x_var.value
            self.followers[i].decision = x_var.value
        diff = np.sqrt(np.sum(np.power(self.leader.decision - p_prev, 2) + np.power(x_prev - x, 2)))
        return diff

    def prox_iterations(self):
        for i in range(self.prox_max_iter):
            print("ITERATION :", i+1)
            self.print_information()
            diff = self.prox_one_iteration()
            print("diff :", diff)
            if i % 5 == 0:
                self.save_data()
            if np.abs(diff) < self.prox_eps:
                break
        ve = self.compute_followers_ve()
        self.followers_decision_history += [ve]
        for i in range(self.ev_num):
            self.followers[i].update(ve[i])
        return 0


class Leader:
    def __init__(self, initial_price, p_min, p_max, p_avg):
        self.decision = initial_price
        self.p_min = p_min
        self.p_max = p_max
        self.p_avg = p_avg
        self.time = len(self.decision)

    def update(self, gradient, step_size):
        self.decision = self.decision + step_size * gradient
        p = cp.Variable(self.time)
        obj = cp.Minimize(cp.sum(cp.power(p - self.decision, 2)))
        const = [self.p_min*np.ones(self.time) <= p, p <= self.p_max*np.ones(self.time), cp.sum(p) == self.time * self.p_avg]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.decision = p.value
        # 필요한 경우 projection 도 진행


class Follower:
    def __init__(self, initial_charging, x_min, x_max, require_load):
        self.decision = initial_charging
        self.ev_min = x_min
        self.ev_max = x_max
        self.ev_load = require_load

    def update(self, new_decision):
        self.decision = new_decision