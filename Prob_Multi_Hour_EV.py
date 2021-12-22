import numpy as np
import cvxpy as cp


class OneTimeEV: # 인자로 유저 수, bn list, sn list
    step_size = 0.001 # Gradient descent step size
    max_iter = 1000
    eps = 1e-10

    p_init = 1 # Initial price of the leader
    x_init = 0 # Initial charging of the followers
    followers = []
    leader_decision_history = []
    followers_decision_history = []

    def __init__(self, users, battery_capacity, satisfaction_coefficient, energy_limit):
        self.bats = battery_capacity
        self.sats = satisfaction_coefficient
        self.ev_num = users
        self.energy_limit = energy_limit
        self.leader = Leader(self.p_init)
        for i in range(self.ev_num):
            self.followers += [Follower(self.x_init, battery_capacity[i], satisfaction_coefficient[i])]

    def compute_followers_ve(self): # and return
        x = np.zeros(self.ev_num)
        b_sum = 0
        s_sum = 0
        for ev in self.followers:
            b_sum += ev.bat/ev.sat
            s_sum += 1/ev.sat
        if b_sum - self.leader.decision*s_sum <= self.energy_limit:
            for i in range(self.ev_num):
                ev = self.followers[i]
                x[i] = (ev.bat - self.leader.decision)/ev.sat
        else:
            for i in range(self.ev_num):
                ev = self.followers[i]
                x[i] = (ev.bat - (b_sum - self.energy_limit)/s_sum)/ev.sat
        return x

    def print_information(self):
        print("Leader Decision :", self.leader.decision)
        x = []
        for f in self.followers:
            x += [f.decision]
        print("EVs Decison     :", x)

    def inequality_const_value(self):
        v = np.zeros(3)
        for f in self.followers:
            v[0] += f.decision
        v[0] -= self.energy_limit
        v[1] = v[0]
        v[2] = self.followers[0].sat * self.followers[0].decision + self.leader.decision - self.followers[0].bat
        return v

    def is_active_inequality_const(self):
        v = self.inequality_const_value()
        #x = np.zeros(len(v))
        #for i in range(len(x)):
        #    if np.abs(v[i]) <= 1e-2:
        #        x[i] = 1
        x = np.isin(v, [0])
        return x

    def compute_leader_gradient(self): # and return
        self.print_information()
        ineq_const_value = self.inequality_const_value()
        print(ineq_const_value)
        active_g = self.is_active_inequality_const()
        print(active_g)
        lagrange_value = - ineq_const_value[2] # 1e-5

        dyh = np.zeros((self.ev_num+1, 2*self.ev_num+1))
        dyh[:self.ev_num, :self.ev_num] = np.diag(self.sats)
        dyh[:self.ev_num, 2 * self.ev_num] = np.ones(self.ev_num)
        dyh[self.ev_num, self.ev_num:2 * self.ev_num] = lagrange_value * np.ones(self.ev_num)
        dyh[self.ev_num, 2 * self.ev_num] = ineq_const_value[0]

        dyg = np.zeros((3, 2*self.ev_num+1))
        dyg[0, :self.ev_num] = np.ones(self.ev_num)
        dyg[1, self.ev_num:2 * self.ev_num] = np.ones(self.ev_num)
        dyg[2, 2*self.ev_num] = -1

        A = dyh
        for i in range(len(active_g)):
            if active_g[i]:
                print(i)
                A = np.vstack((A, dyg[i]))
        print(A)
        AAT_inv = np.linalg.inv(A@A.T)

        dyf = np.zeros(2*self.ev_num+1)
        i = 0
        for f in self.followers:
            dyf[i] = 2*f.sat * f.decision + self.leader.decision - f.sat * f.decision - f.bat
            dyf[self.ev_num + i] = - f.sat * f.decision
            i += 1

        lambda_values = AAT_inv@A@dyf.T
        print(lambda_values)
        dxyf = np.ones((2*self.ev_num+1, 1))
        dxyf[self.ev_num:2*self.ev_num] = - np.ones((self.ev_num, 1))
        dxyf[2*self.ev_num, 0] = 0

        dxyh = np.zeros((self.ev_num+1, 2*self.ev_num+1, 1))
        dxyg = np.zeros((3, 2*self.ev_num+1, 1))

        dxy_h_wave = dxyh
        for i in range(len(active_g)):
            if active_g[i]:
                dxy_h_wave = np.vstack((dxy_h_wave, dxyg))
        for i in range(len(lambda_values)):
            dxy_h_wave[i] *= lambda_values[i]

        B = dxyf - np.sum(dxy_h_wave, axis=0)

        dxh = np.ones((self.ev_num+1, 1))
        dxh[self.ev_num, 0] = 0
        dxg = np.zeros((3, 1))

        C = dxh
        for i in range(len(active_g)):
            if active_g[i]:
                C = np.vstack((C, dxg[i]))

        dyyf = np.zeros((2*self.ev_num+1, 2*self.ev_num+1))
        dyyf[:2*self.ev_num, :2*self.ev_num] = np.kron(np.array([[2, -1], [-1, 0]]), np.diag(self.sats))

        dyyh = np.zeros((self.ev_num+1, 2*self.ev_num+1, 2*self.ev_num+1))
        dyyh[self.ev_num, self.ev_num:2*self.ev_num, 2*self.ev_num] = np.ones(self.ev_num)
        dyyh[self.ev_num, 2*self.ev_num, self.ev_num:2*self.ev_num] = np.ones(self.ev_num)
        dyyg = np.zeros((3, 2*self.ev_num+1, 2*self.ev_num+1))

        dyy_h_wave = dyyh
        for i in range(len(active_g)):
            if active_g[i]:
                dyy_h_wave = np.vstack((dyy_h_wave, dyyg[i].reshape((1, 2*self.ev_num+1, 2*self.ev_num+1))))
        for i in range(len(lambda_values)):
            dyy_h_wave[i] *= lambda_values[i]

        H = dyyf - np.sum(dyy_h_wave, axis=0)

        H_inv = np.linalg.inv(H)

        dy = H_inv@A.T@np.linalg.inv(A@H_inv@A.T)@(A@H_inv@B-C)-H_inv@B

        dxj = 0
        for f in self.followers:
            dxj += f.decision

        dyj = np.zeros((1, 2*self.ev_num+1))
        dyj[0, :self.ev_num] = self.leader.decision
        dj = dxj + dyj@dy
        return dj

    def one_iteration(self): # 현재의 leader action에 대해서 follower들의 ve를 구한 후 gradient를 구해서 leader update
        self.leader_decision_history += [self.leader.decision]
        ve = self.compute_followers_ve()
        self.followers_decision_history += [ve]
        for i in range(self.ev_num):
            self.followers[i].update(ve[i])
        grad = self.compute_leader_gradient()
        self.leader.update(grad, self.step_size)
        next_leader_decision = self.leader.decision
        diff = next_leader_decision - self.leader_decision_history[-1]
        return diff

    def iterations(self):
        for _ in range(self.max_iter):
            self.print_information()
            diff = self.one_iteration()
            if np.abs(diff) < self.eps:
                break
        return 0

    def leader_utility(self):
        s = 0
        for i in range(self.ev_num):
            s += self.followers[i].decision
        return self.leader.decision * s

    def followers_utility(self):
        u = np.zeros(self.ev_num)
        for i in range(self.ev_num):
            f = self.followers[i]
            u[i] = (f.bat - self.leader.decision) * f.decision - 0.5 * f.sat * f.decision**2
        return u

    def save_data(self, filename):
        np.save(filename, [self.leader_decision_history, self.followers_decision_history])
        return 0


class Leader:
    def __init__(self, initial_price):
        self.decision = initial_price

    def update(self, gradient, step_size):
        self.decision = self.decision + step_size * gradient
        # 필요한 경우 projection 도 진행


class Follower:
    def __init__(self, initial_charging, battery_capacity, satisfaction_coefficient):
        self.decision = initial_charging
        self.bat = battery_capacity
        self.sat = satisfaction_coefficient

    def update(self, new_decision):
        self.decision = new_decision


prob = OneTimeEV(3, [2, 4, 6], [1, 1.5, 2], 20)
prob.iterations()


