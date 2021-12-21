import numpy as np
import cvxpy as cp
#from inheritance import *


class SymmetricStackelbergGame: # 인자로 시간/액티브,패시브 유저 수/각 유저들의 시간별 전기 요구량/생산량, parameter
    def __init__(self, _time, active_user_number, passive_user_number, active_user_load, passive_user_load,
                 _alpha, _beta_s, _beta_b):
        self._time = _time
        self._active = active_user_number
        self._passive = passive_user_number
        self.alpha = _alpha
        self.beta_s = _beta_s
        self.beta_b = _beta_b
        self.active_users = []
        self.passive_users = []
        self.leader = Leader(self._time)
        for i in range(self._active):
            self.active_users += [Follower(self._time, True, active_user_load[i])]
        for i in range(self._passive):
            self.passive_users += [Follower(self._time, False, passive_user_load[i])]

    def time(self):
        return self._time()

    def users(self):
        return [self._active, self._passive]

    def leader_utility(self):
        return self.leader.buy_price

    def follower_utility(self):
        return self.leader.buy_price

    def optimize(self, r):
        p = cp.Variable(self.size)
        obj = -cp.sum(cp.power(p, 2))
        const = [p <= 1, p >= 0]
        prob = cp.Problem(cp.Maximize(obj), const)
        result = prob.solve(solver='ECOS')
        print(p.value)
        print(obj.value)
        return p.value


class Leader:
    def __init__(self, _time, _buy_price=None, _sell_price=None):
        self._time = _time
        if _buy_price is None:
            self.buy_price = 2*np.ones(self._time)
        else:
            self.buy_price = _buy_price
        if _sell_price is None:
            self.sell_price = 2 * np.ones(self._time)
        else:
            self.sell_price = _sell_price

    def update(self, _buy_price, _sell_price):
        self.buy_price = _buy_price
        self.sell_price = _sell_price

    def projected_update(self, _buy_price, _sell_price):
        pb = cp.Variable(self._time)
        ps = cp.Variable(self._time)
        obj = cp.sum(cp.power(pb - _buy_price, 2)) + cp.sum(cp.power(ps - _sell_price, 2))
        const = [pb >= ps]
        prob = cp.Problem(cp.Minimize(obj), const)
        result = prob.solve(solver='ECOS')

        self.buy_price = pb.value
        self.sell_price = ps.value
        print("Given buy price :", _buy_price)
        print("Projected price :", self.buy_price)
        print("Given buy price :", _sell_price)
        print("Projected sell price :", self.sell_price)


class Follower: # load는 필요 전기량 - 생산량으로 음수일 수 있다. 음수인 경우 passive user는 전기를 팔거나 저장하지 못하고 버린다.
    def __init__(self, _time, is_active, require): # 인자로 Active/Passive 여부, 시간별 전기 요구량/생산량
        self._time = _time
        self.active = is_active
        self.require = require
        if self.active:
            self.grid_buy = np.maximum(require, np.zeros(self._time))
            self.ess_buy = np.zeros(self._time)
            self.ess_sell = self.grid_buy - self.ess_buy - self.require
            self.trash = np.zeros(self._time)
        if not self.active:
            self.grid_buy = np.maximum(require, np.zeros(self._time))
            self.ess_buy = np.zeros(self._time)
            self.ess_sell = np.zeros(self._time)
            self.trash = self.grid_buy - self.ess_buy - self.require

    def update(self, _grid_buy, _ess_buy, _ess_sell, _trash=None):
        self.grid_buy = _grid_buy
        self.ess_buy = _ess_buy
        self.ess_sell = _ess_sell
        if _trash is not None:
            self.trash = _trash