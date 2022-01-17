import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt

class EET_Parameters:
    def __init__(self, params):
        [active_users, passive_users, times, alpha, beta_s, beta_b, q_max, q_init, c_max, c_min, grid_price, tax, soh, active_require_loads, passive_require_loads] = params
        self.active_num = active_users
        self.passive_num = passive_users
        self.time = times
        self.alpha = alpha
        self.beta_s = beta_s
        self.beta_b = beta_b
        self.q_max = q_max
        self.q_init = q_init
        self.c_max = c_max
        self.c_min = c_min
        self.grid_price = grid_price
        self.tax = tax
        self.soh = soh
        self.active_load = active_require_loads # active user x times
        self.passive_load = np.maximum(passive_require_loads, np.zeros((self.passive_num, self.time))) # passive user x times


class EET_Game(EET_Parameters):
    step_size = 0.1 # Gradient descent step size
    max_iter = 10000
    eps = 1e-10
    ve_step_size = 0.001#0.0145
    ve_max_iter = 20000
    ve_eps = 1e-8
    approx_ve_eps = 1e-5
    active_epsilon = 1e-4
    prox_gamma = 10
    prox_eps = 1e-6
    prox_max_iter = 10000

    active_followers = []
    passive_followers = []
    leader_decision_history = []
    followers_decision_history = []
    leader_utility_history = []
    par_history = []

    def __init__(self, params, filename=None):
        super().__init__(params)
        self.p_init = np.kron(np.array([[1], [2]]), np.ones(self.time))  # Initial price of the leader
        self.x_init = np.zeros((self.active_num, 2, self.time)) # Initial action of the followers
        self.x_init[:, 0, :] = -np.minimum(np.zeros((self.active_num, self.time)), self.active_load)
        self.leader = Leader(self.p_init)
        self.filename = filename
        for i in range(self.active_num):
            self.active_followers += [Follower(self.x_init[i], self.active_load[i], True)]
        for i in range(self.passive_num):
            self.passive_followers += [Follower(np.zeros((2, self.time)), self.passive_load[i], False)]

    def compute_followers_ve(self): # and return
        print("VE")
        x_cur = np.zeros((self.active_num, 2, self.time))

        x_s_cur = np.zeros((self.active_num, self.time))
        x_b_cur = np.zeros((self.active_num, self.time))
        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            x_s_cur[i] = self.active_followers[i].x_s
            x_b_cur[i] = self.active_followers[i].x_b
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        p = self.leader.decision

        x_cur[:, 0, :] = x_s_cur
        x_cur[:, 1, :] = x_b_cur
        x_next = x_cur

        tmp_step_size = self.ve_step_size
        ve_err_prev = -1e10
        for iter in range(self.ve_max_iter):
            #print("x_cur :", np.sum(x_cur, axis=0))
            Fx = np.zeros((self.active_num, 2, self.time))
            Fx[:, 0, :] = -np.kron(np.ones((self.active_num, 1)), p[0]) + self.grid_price*(np.kron(np.ones((self.active_num, 1)), np.sum(l_cur, axis=0)+np.sum(l_passive, axis=0))+l_cur) + \
                          self.soh*(np.kron(np.ones((self.active_num, 1)), np.sum(x_s_cur, axis=0))+x_s_cur) # grid price가 시간마다 다를 경우 고려 x된 implementation
            Fx[:, 1, :] = np.kron(np.ones((self.active_num, 1)), p[1]) - self.grid_price * (
                        np.kron(np.ones((self.active_num, 1)),
                                np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)) + l_cur) + \
                          self.soh * (np.kron(np.ones((self.active_num, 1)), np.sum(x_b_cur,
                                                                                    axis=0)) + x_b_cur)  # grid price가 시간마다 다를 경우 고려 x된 implementation
            x_next = x_cur - tmp_step_size * Fx

            x_s_proj = cp.Variable((self.active_num, self.time))
            x_b_proj = cp.Variable((self.active_num, self.time))
            obj = cp.Minimize(cp.sum(cp.power(x_next[:, 0, :] - x_s_proj, 2) + cp.power(x_next[:, 1, :] - x_b_proj, 2)))
            ess_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a < b else np.power(self.alpha, a - b)),
                                         (self.time, self.time), dtype=float)
            ess_init_vector = self.alpha * ess_matrix[:, 0]

            q_ess = self.q_init * ess_init_vector + ess_matrix @ (self.beta_s*cp.sum(x_s_proj, axis=0) - self.beta_b*cp.sum(x_b_proj, axis=0))

            const = []
            const += [q_ess >= 0, q_ess <= self.q_max]
            const += [cp.sum(x_s_proj, axis=0) <= self.c_max, cp.sum(x_b_proj, axis=0) <= self.c_min]
            const += [x_s_proj >= 0, x_b_proj >= 0, x_s_proj - x_b_proj + self.active_load >= 0]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            #print("RESULT :", result)
            #print("SOC :", q_ess.value)

            x_s_next = x_s_proj.value
            x_b_next = x_b_proj.value
            l_next = x_s_next - x_b_next + self.active_load
            x_next[:, 0, :] = x_s_next
            x_next[:, 1, :] = x_b_next

            for i in range(self.active_num):
                self.active_followers[i].update(x_next[i])
            if (iter+1)%200 == 0:
                print("ITER", iter+1, "VE GAP :", np.sqrt(np.sum(np.power(x_cur - x_next, 2))))
                #self.print_information()
            if np.sqrt(np.sum(np.power(x_cur - x_next, 2))) <= self.ve_eps and iter >= 1000-1 :
                break
            if iter==-1:# or (iter+1)%2000 == 0:# and abs(result) < 1:
                x_s_check = cp.Variable((self.active_num, self.time))
                x_b_check = cp.Variable((self.active_num, self.time))
                Fx = np.zeros((self.active_num, 2, self.time))
                Fx[:, 0, :] = -np.kron(np.ones((self.active_num, 1)), p[0]) + self.grid_price * (
                            np.kron(np.ones((self.active_num, 1)),
                                    np.sum(l_next, axis=0) + np.sum(l_passive, axis=0)) + l_next) + \
                              self.soh * (np.kron(np.ones((self.active_num, 1)), np.sum(x_s_next,
                                                                                        axis=0)) + x_s_next)  # grid price가 시간마다 다를 경우 고려 x된 implementation
                Fx[:, 1, :] = np.kron(np.ones((self.active_num, 1)), p[1]) - self.grid_price * (
                        np.kron(np.ones((self.active_num, 1)),
                                np.sum(l_next, axis=0) + np.sum(l_passive, axis=0)) + l_next) + \
                              self.soh * (np.kron(np.ones((self.active_num, 1)), np.sum(x_b_next,
                                                                                        axis=0)) + x_b_next)  # grid price가 시간마다 다를 경우 고려 x된 implementation

                obj = cp.Minimize(cp.sum(cp.multiply(Fx[:, 0, :], x_s_check - x_s_next) + cp.multiply(Fx[:, 1, :], x_b_check - x_b_next)))
                ess_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a < b else np.power(self.alpha, a - b)),
                                             (self.time, self.time), dtype=float)
                ess_init_vector = self.alpha * ess_matrix[:, 0]

                q_ess_check = self.q_init * ess_init_vector + ess_matrix @ (
                            self.beta_s * cp.sum(x_s_check, axis=0) - self.beta_b * cp.sum(x_b_check, axis=0))

                const = []
                const += [q_ess_check >= 0, q_ess_check <= self.q_max]
                const += [cp.sum(x_s_check, axis=0) <= self.c_max, cp.sum(x_b_check, axis=0) <= self.c_min]
                const += [x_s_check >= 0, x_b_check >= 0, x_s_check - x_b_check + self.active_load >= 0]
                prob = cp.Problem(obj, const)
                result = prob.solve(solver='ECOS')

                #if ve_err_prev > result:
                #    tmp_step_size *= 0.9
                #    print("CURRENT STEP SIZE :", tmp_step_size)
                #    if tmp_step_size < 1e-5:
                #        print("LAST VE CHECKING :", result)
                #        break
                #    continue
                #print(ve_err_prev, result, self.approx_ve_eps)
                if ((ve_err_prev > result) and (result < 1e-4)) or abs(result) < self.approx_ve_eps:
                    print("ITER :", iter + 1, "LAST VE CHECKING :", result)
                    break
                else:
                    print("ITER :", iter + 1, "VE CHECKING :", result)
                    ve_err_prev = result
                #ve_err_hist[:3] = ve_err_hist[1:]
                #ve_err_hist[3] = result
            x_cur = x_next
            x_s_cur = x_s_next
            x_b_cur = x_b_next
            l_cur = l_next

        self.print_information()
        print("VE Done")
        return x_next

    def print_information(self):
        print("Leader Decision :", self.leader.decision)
        print("Leader Utility  :", self.leader_utility())
        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l
        grid_load = np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)
        print("PAR :", np.max(grid_load)/np.average(grid_load))
        #plt.figure()
        #plt.plot(grid_load, label='Grid load')
        #plt.plot(np.sum(self.active_load, axis=0) + np.sum(self.passive_load, axis=0), label='Require Load')
        #plt.legend()
        #plt.show()
        print()
        #x = []
        #for f in self.active_followers:
        #    x += [f.decision]
        #for i in range(self.ev_num):
        #    continue#print("EVs Decison     :", x[i])

    def inequality_const_value(self):
        v_t_g = np.zeros((6, self.time))
        v_t_lamb = np.zeros((6, self.time))
        v_nt_g = np.zeros((3, self.time, self.active_num))
        v_nt_lamb = np.zeros((3, self.time, self.active_num))

        x_s_cur = np.zeros((self.active_num, self.time))
        x_b_cur = np.zeros((self.active_num, self.time))
        l_cur = np.zeros((self.active_num, self.time))
        for i in range(self.active_num):
            x_s_cur[i] = self.active_followers[i].x_s
            x_b_cur[i] = self.active_followers[i].x_b
            l_cur[i] = self.active_followers[i].l

        ess_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a < b else np.power(self.alpha, a - b)),
                                     (self.time, self.time), dtype=float)
        ess_init_vector = self.alpha * ess_matrix[:, 0]
        q_ess = self.q_init * ess_init_vector + ess_matrix @ (
                    self.beta_s * np.sum(x_s_cur, axis=0) - self.beta_b * np.sum(x_b_cur, axis=0))

        v_t_g[0, :] = - q_ess
        v_t_g[1, :] = q_ess - self.q_max * np.ones(self.time)
        v_t_g[2, :] = - np.sum(x_s_cur, axis=0)
        v_t_g[3, :] = np.sum(x_s_cur, axis=0) - self.c_max * np.ones(self.time)
        v_t_g[4, :] = - np.sum(x_b_cur, axis=0)
        v_t_g[5, :] = np.sum(x_b_cur, axis=0) - self.c_min * np.ones(self.time)

        v_nt_g[0, :, :] = - x_s_cur.T
        v_nt_g[1, :, :] = - x_b_cur.T
        v_nt_g[2, :, :] = - l_cur.T

        f = lambda x: np.abs(x) <= self.active_epsilon

        v_t_lamb += f(v_t_g)
        v_nt_lamb += f(v_nt_g)

        return [v_t_g, v_nt_g, v_t_lamb, v_nt_lamb]

    def is_active_inequality_const(self):
        [v_t_g, v_nt_g, v_t_lamb, v_nt_lamb] = self.inequality_const_value()
        f = lambda x: np.abs(x) <= self.active_epsilon
        active = np.hstack((f(v_t_g).reshape(-1), f(v_nt_g).reshape(-1), f(v_t_lamb).reshape(-1), f(v_nt_lamb).reshape(-1)))
        #print("ACTIVE", len(active))
        #print(np.sum(active))
        #print(active)
        return active

    def compute_leader_gradient(self): # and return
        print("Gradient")
        active = self.is_active_inequality_const()
        Dxh = np.zeros((self.time*(6+5*self.active_num), 2*self.time))
        Dxh[:2*self.time*self.active_num, :] = np.kron(np.kron(np.array([[-1, 0], [0, 1]]), np.eye(self.time)), np.ones((self.active_num, 1)))

        Dxh_wave = Dxh
        #print(Dxh_wave.shape)

        help_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a > b else np.power(self.alpha, b-a)),
                                     (self.time, self.time), dtype=float)

        Dyh = np.zeros((2*self.active_num*self.time, self.time*(6+5*self.active_num)))
        Dyh[:, :2*self.active_num*self.time] = np.kron(np.kron(np.array([[self.grid_price + self.soh, -self.grid_price], [-self.grid_price, self.grid_price + self.soh]]),
                                                       np.eye(self.time)), np.ones((self.active_num, self.active_num))+np.eye(self.active_num))
        Dyh[:, 2*self.active_num*self.time:2*self.active_num*self.time+2*self.time] = np.kron(np.array([[-self.beta_s, self.beta_s], [self.beta_b, -self.beta_b]]),
                                                                                              np.kron(help_matrix, np.ones((self.active_num, 1))))
        Dyh[:, 2 * self.active_num * self.time + 2 * self.time : 2 * self.active_num * self.time + 6 * self.time] = np.kron(np.array([[-1, 1, 0, 0], [0, 0, -1, 1]]),
                                                                                                                            np.kron(np.eye(self.time), np.ones((self.active_num, 1))))
        Dyh[:, 2 * self.active_num * self.time + 6 * self.time:] = np.kron(np.array([[-1, 0, -1], [0, -1, 1]]), np.eye(self.time*self.active_num))

        Dyg = np.zeros((2*self.time*(6+3*self.active_num), self.time*(6+5*self.active_num)))
        Dyg[self.time*(6+3*self.active_num):, 2*self.time*self.active_num:] = - np.eye(self.time*(6+3*self.active_num))
        Dyg[:2*self.time, :2*self.time*self.active_num] = np.kron(np.array([[self.beta_s, -self.beta_b], [-self.beta_s, self.beta_b]]),
                                                                  np.kron(help_matrix.T, np.ones(self.active_num)))
        Dyg[2 * self.time: 6* self.time, :2 * self.time * self.active_num] = np.kron(np.array([[-1,0],[1,0],[0,-1],[0,1]]),
                                                                                     np.kron(np.eye(self.time), np.ones(self.active_num)))
        Dyg[6 * self.time:self.time*(6+3*self.active_num), :2 * self.time * self.active_num] = np.kron(np.array([[-1, 0], [0, -1], [1, -1]]),
                                                                                                       np.eye(self.time*self.active_num))
        Dyh_wave = Dyh
        for i in range(len(active)):
            if active[i]:
                Dyh_wave = np.vstack((Dyh_wave, Dyg[i]))
        print(Dyh_wave.shape, Dxh_wave.shape)
        #"""
        Dy_var = cp.Variable((self.time*(6+5*self.active_num), 2*self.time))
        #print("SHAPE :", Dxh_wave.shape, Dyh_wave.shape, Dy_var.shape)
        obj = cp.Minimize(1)
        const = [Dyh_wave@Dy_var + Dxh_wave == 0]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        dy = Dy_var.value
        #"""
        #dy = -np.linalg.inv(Dyh_wave)@Dxh_wave

        p = self.leader.decision

        dxj = - 2 * self.tax * p.reshape(-1)

        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        grid_load = np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)
        dyj = np.kron(np.array([1, -1]), -2*self.grid_price*np.kron(grid_load, np.ones(self.active_num)))
        #print("DyJ :", dyj)
        dj = dxj + dyj@dy[:2*self.time*self.active_num, :]
        print("DJ :", dj)
        return dj

    def compute_leader_gradient_large_size(self): # and return
        print("Gradient")
        active = self.is_active_inequality_const()
        Dxh = np.zeros((self.time*(6+5*self.active_num), 2*self.time))
        Dxh[:2*self.time*self.active_num, :] = np.kron(np.kron(np.array([[-1, 0], [0, 1]]), np.eye(self.time)), np.ones((self.active_num, 1)))

        Dxh_wave = Dxh
        #print(Dxh_wave.shape)

        help_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a > b else np.power(self.alpha, b-a)),
                                     (self.time, self.time), dtype=float)

        Dyh = np.zeros((2*self.active_num*self.time, self.time*(6+5*self.active_num)))
        Dyh[:, :2*self.active_num*self.time] = np.kron(np.kron(np.array([[self.grid_price + self.soh, -self.grid_price], [-self.grid_price, self.grid_price + self.soh]]),
                                                       np.eye(self.time)), np.ones((self.active_num, self.active_num))+np.eye(self.active_num))
        Dyh[:, 2*self.active_num*self.time:2*self.active_num*self.time+2*self.time] = np.kron(np.array([[-self.beta_s, self.beta_s], [self.beta_b, -self.beta_b]]),
                                                                                              np.kron(help_matrix, np.ones((self.active_num, 1))))
        Dyh[:, 2 * self.active_num * self.time + 2 * self.time : 2 * self.active_num * self.time + 6 * self.time] = np.kron(np.array([[-1, 1, 0, 0], [0, 0, -1, 1]]),
                                                                                                                            np.kron(np.eye(self.time), np.ones((self.active_num, 1))))
        Dyh[:, 2 * self.active_num * self.time + 6 * self.time:] = np.kron(np.array([[-1, 0, -1], [0, -1, 1]]), np.eye(self.time*self.active_num))

        Dyg = np.zeros((2*self.time*(6+3*self.active_num), self.time*(6+5*self.active_num)))
        Dyg[self.time*(6+3*self.active_num):, 2*self.time*self.active_num:] = - np.eye(self.time*(6+3*self.active_num))
        Dyg[:2*self.time, :2*self.time*self.active_num] = np.kron(np.array([[self.beta_s, -self.beta_b], [-self.beta_s, self.beta_b]]),
                                                                  np.kron(help_matrix.T, np.ones(self.active_num)))
        Dyg[2 * self.time: 6* self.time, :2 * self.time * self.active_num] = np.kron(np.array([[-1,0],[1,0],[0,-1],[0,1]]),
                                                                                     np.kron(np.eye(self.time), np.ones(self.active_num)))
        Dyg[6 * self.time:self.time*(6+3*self.active_num), :2 * self.time * self.active_num] = np.kron(np.array([[-1, 0], [0, -1], [1, -1]]),
                                                                                                       np.eye(self.time*self.active_num))
        Dyh_wave = Dyh
        for i in range(len(active)):
            if active[i]:
                Dyh_wave = np.vstack((Dyh_wave, Dyg[i]))
        print(Dyh_wave.shape, Dxh_wave.shape)
        #"""
        dy = np.zeros((self.time*(6+5*self.active_num), 2*self.time))
        for t in range(2*self.time):
            Dy_var = cp.Variable((self.time*(6+5*self.active_num), 1))
            #print("SHAPE :", Dxh_wave.shape, Dyh_wave.shape, Dy_var.shape)
            obj = cp.Minimize(1)
            const = [Dyh_wave@Dy_var + (Dxh_wave[:,t].reshape(-1,1)) == 0]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')
            dy[:, t] = (Dy_var.value.reshape(-1))
        #"""
        #dy = -np.linalg.inv(Dyh_wave)@Dxh_wave

        p = self.leader.decision

        dxj = - 2 * self.tax * p.reshape(-1)

        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        grid_load = np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)
        dyj = np.kron(np.array([1, -1]), -2*self.grid_price*np.kron(grid_load, np.ones(self.active_num)))
        #print("DyJ :", dyj)
        dj = dxj + dyj@dy[:2*self.time*self.active_num, :]
        print("DJ :", dj)
        return dj

    def one_iteration(self): # 현재의 leader action에 대해서 follower들의 ve를 구한 후 gradient를 구해서 leader update
        self.leader_decision_history += [self.leader.decision]
        ve = self.compute_followers_ve()
        self.followers_decision_history += [ve]
        self.leader_utility_history += [self.leader_utility()]
        self.par_history += [self.par()]
        for i in range(self.active_num):
            self.active_followers[i].update(ve[i])
        grad = self.compute_leader_gradient_large_size()
        self.leader.update(grad, self.step_size)
        next_leader_decision = self.leader.decision
        diff = np.sqrt(np.sum(np.power(next_leader_decision - self.leader_decision_history[-1], 2)))
        return diff

    def iterations(self):
        for i in range(self.max_iter):
            print("ITERATION :", i+1)
            diff = self.one_iteration()
            self.save_data()
            if (i+1)%100 == 0:
                plt.figure()
                plt.plot(-np.array(self.leader_utility_history), label="Leader cost")
                plt.xlabel("Iteration")
                plt.legend()
                plt.show()
                plt.figure()
                plt.plot(self.par_history, label = "par")
                plt.xlabel("Iteration")
                plt.legend()
                plt.show()
            if np.abs(diff) < self.eps:
                break
        return 0

    def leader_utility(self):
        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        grid_load = np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)
        u = - self.grid_price * np.sum(np.power(grid_load, 2)) - self.tax * np.sum(np.power(self.leader.decision, 2))
        return u

    def followers_utility(self):
        x_s = np.zeros((self.active_num, self.time))
        x_b = np.zeros((self.active_num, self.time))
        l = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            x_s[i] = self.active_followers[i].x_s
            x_b[i] = self.active_followers[i].x_b
            l[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        p_s = self.leader.decision[0]
        p_b = self.leader.decision[1]

        c = np.zeros(self.active_num) # cost
        for i in range(self.active_num):
            c[i] += np.sum(np.multiply(p_b, x_b)-np.multiply(p_s, x_s))
            c[i] += self.grid_price*np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(l_passive, axis=0)))
            c[i] += self.soh * np.sum(np.multiply(x_s, np.sum(x_s, axis=0)) + np.multiply(x_b, np.sum(x_b, axis=0)))

        return -c

    def par(self):
        l_cur = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            l_cur[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l
        grid_load = np.sum(l_cur, axis=0) + np.sum(l_passive, axis=0)
        return np.max(grid_load)/np.average(grid_load)

    def save_data(self):
        if self.filename is not None:
            with open(self.filename, "wb") as f:
                pickle.dump(self, f)
        return 0

    def prox_one_iteration(self):
        self.leader_decision_history += [self.leader.decision]
        x_s = np.zeros((self.active_num, self.time))
        x_b = np.zeros((self.active_num, self.time))
        l = np.zeros((self.active_num, self.time))
        l_passive = np.zeros((self.passive_num, self.time))
        for i in range(self.active_num):
            x_s[i] = self.active_followers[i].x_s
            x_b[i] = self.active_followers[i].x_b
            l[i] = self.active_followers[i].l

        for i in range(self.passive_num):
            l_passive[i] = self.passive_followers[i].l

        x = np.zeros((self.active_num, 2, self.time))
        x[:, 0, :] = x_s
        x[:, 1, :] = x_b

        self.followers_decision_history += [x]
        self.leader_utility_history += [self.leader_utility()]
        x_prev = x
        p_prev = self.leader.decision

        p_var = cp.Variable((2, self.time))
        obj = cp.Maximize(- self.tax * cp.sum(cp.power(p_var, 2)) - self.prox_gamma/2*cp.sum(cp.power(p_prev - p_var, 2))) # CONSTANT 부분은 없앰

        const = [p_var[0] <= p_var[1]]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.leader.decision = p_var.value

        for i in range(self.active_num):
            x_s = np.zeros((self.active_num, self.time))
            x_b = np.zeros((self.active_num, self.time))
            l = np.zeros((self.active_num, self.time))
            l_passive = np.zeros((self.passive_num, self.time))
            for i in range(self.active_num):
                x_s[i] = self.active_followers[i].x_s
                x_b[i] = self.active_followers[i].x_b
                l[i] = self.active_followers[i].l

            for i in range(self.passive_num):
                l_passive[i] = self.passive_followers[i].l

            p_s = self.leader.decision[0]
            p_b = self.leader.decision[1]

            x_s_var = cp.Variable(self.time)
            x_b_var = cp.Variable(self.time)
            l_var = cp.Variable(self.time)
            grid_load_except_i = np.sum(l, axis=0) + np.sum(l_passive, axis=0) - l[i]
            x_s_sum_except_i = np.sum(x_s, axis=0) - x_s[i]
            x_b_sum_except_i = np.sum(x_b, axis=0) - x_b[i]
            obj = cp.Minimize(cp.sum(cp.multiply(p_b,x_b_var)) - cp.sum(cp.multiply(p_s,x_s_var)) + self.grid_price * (cp.sum(cp.power(l_var, 2)+cp.multiply(grid_load_except_i, l_var)))
                              + self.soh * (cp.sum(cp.power(x_s_var, 2) + cp.multiply(x_s_sum_except_i,x_s_var)+cp.power(x_b_var, 2) + cp.multiply(x_b_sum_except_i,x_b_var)))
                              + self.prox_gamma/2*cp.sum(cp.power(x_s_var - x[i, 0, :], 2)+cp.power(x_b_var - x[i, 1, :], 2)))

            ess_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a < b else np.power(self.alpha, a - b)),
                                         (self.time, self.time), dtype=float)
            ess_init_vector = self.alpha * ess_matrix[:, 0]
            print(ess_matrix.shape)
            print(cp.sum(x_s_sum_except_i + x_s_var, axis=0))
            q_ess = self.q_init * ess_init_vector + ess_matrix @ (self.beta_s * (x_s_sum_except_i + x_s_var) - self.beta_b * (x_b_sum_except_i + x_b_var))

            const = [x_s_var >= 0, x_b_var >= 0, l_var >= 0]
            const += [x_s_sum_except_i + x_s_var <= self.c_max]
            const += [x_b_sum_except_i + x_b_var <= self.c_min]
            const += [q_ess >= 0, q_ess <= self.q_max]
            prob = cp.Problem(obj, const)
            result = prob.solve(solver='ECOS')

            new = np.zeros((2, self.time))
            new[0, :] = x_s_var.value
            new[1, :] = x_b_var.value
            self.active_followers[i].update(new)
        diff = np.sqrt(np.sum(np.power(self.leader.decision - p_prev, 2) + np.power(x_prev - x, 2)))
        return diff

    def prox_iterations(self):
        self.leader.decision = np.zeros((2, self.time))
        """
        for i in range(self.prox_max_iter):
            print("ITERATION :", i+1)
            self.print_information()
            diff = self.prox_one_iteration()
            print("diff :", diff)
            if i % 5 == 0:
                self.save_data()
            if np.abs(diff) < self.prox_eps:
                break
        """
        self.leader_decision_history += [self.leader.decision]
        ve = self.compute_followers_ve()
        #for i in range(self.active_num):
        #    self.active_followers[i].update(ve[i])
        self.followers_decision_history += [ve]
        self.leader_utility_history += [self.leader_utility()]
        self.par_history += [self.par()]
        self.save_data()

        return 0


class Leader:
    def __init__(self, initial_price):
        self.decision = initial_price
        self.time = self.decision.shape[-1]

    def update(self, gradient, step_size):
        self.decision = self.decision + step_size * gradient.reshape(2, -1)
        p = cp.Variable((2, self.time))
        obj = cp.Minimize(cp.sum(cp.power(p - self.decision, 2)))
        const = [0<=p[0], p[0] <= p[1]]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.decision = p.value
        # 필요한 경우 projection 도 진행


class Follower:
    def __init__(self, initial_charging, require_load, is_active):
        self.is_active = is_active
        self.decision = initial_charging
        self.require_load = require_load
        self.x_s = self.decision[0]
        self.x_b = self.decision[1]
        self.l = self.x_s - self.x_b + self.require_load

    def update(self, new_decision):
        self.decision = new_decision
        self.x_s = self.decision[0]
        self.x_b = self.decision[1]
        self.l = self.x_s - self.x_b + self.require_load