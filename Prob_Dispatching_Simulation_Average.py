from Prob_Dispatching import *


def simulation(num_ev, num_station, num_experiment):
    for exp in range(1, num_experiment+1):
        ev_ = num_ev
        sts_ = num_station
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
        coef = 0.2*np.random.random((ev_, 3)) + 0.2
        #print(len(ev_loc), len(station_loc), len(elec_max), len(ev_max))
        parameter = [ev_, sts_, ev_loc, station_loc, 1, 20, target_ev, load, coef, elec_max, ev_max]
        prob = Dispatching(parameter, "results/average/"+str(ev_)+"ev_"+str(sts_)+"st"+"/env_"+str(exp)+".pkl")
        prob.grad_history.initialize()
        prob.heur_history.initialize()
        prob.prox_history.initialize()
        prob.initialize_action()
        prob.grad_iterations()
        #prob.initialize_action()
        #prob.heur_iterations()
        prob.initialize_action()
        prob.prox_iterations()

simulation(50, 5, 30)