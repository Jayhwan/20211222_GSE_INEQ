from Prob_Dispatching import *

ev_ = 200
sts_ = 10
ev_loc = []
station_loc = []
for i in range(ev_):
    ev_loc += [np.array([5*np.random.random(), 5*np.random.random()])]

for i in range(sts_):
    station_loc += [np.array([5*np.random.random(), 5*np.random.random()])]
#for i in range(25):
#    ev_loc += [np.array([5*np.random.random(), 5*np.random.random()])]
target_ev = np.random.random(sts_)
target_ev = ev_*target_ev/np.sum(target_ev)

load = 0.8*np.random.random(ev_)+0.2
elec_max = 3*np.random.random(sts_)+ev_/sts_
ev_max = ev_*np.ones(sts_)
#print(len(ev_loc), len(station_loc), len(elec_max), len(ev_max))
parameter = [ev_, sts_, ev_loc, station_loc, 1, 20, target_ev, load, np.ones((ev_, 3))/3, elec_max, ev_max]
prob = Dispatching(parameter, "test_dispatching.pkl")
prob.draw_map()
prob.grad_one_iteration()