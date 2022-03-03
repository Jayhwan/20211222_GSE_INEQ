from Prob_Dispatching import *

#with open("Dispatching/파라미터저장용.pkl", 'rb') as f:
#    prob_ = pickle.load(f)

#evs_load = prob_.evs_load
with open("Dispatching/Dispatching_main.pkl", 'rb') as f:
    prob = pickle.load(f)

prob.change_filename("Dispatching/Dispatching_main_prox_test.pkl")
prob.initialize_action()
prob.prox_history = History()
prob.heur_history = History()
prob.grad_history = History()

#prob.evs_load = evs_load
print(prob.evs, prob.stations)
print(prob.evs_loc, prob.sts_loc)
print(prob.p_min, prob.p_max)
print(prob.target_evs)
print(prob.evs_load)
print(prob.evs_priority)
print(prob.max_elec)
print(prob.max_evs)
#print(prob.leader.decision())
print(prob.leader_action())
print(prob.followers_action())
print("###########")
prob.heur_history = History()
prob.leader.update_direct((prob.p_min + prob.p_max)/2 * np.ones(prob.stations))
prob.update_followers(np.ones((prob.evs, prob.stations))/prob. stations)
print("###########")
#prob.grad_history = grad_history
#prob.prox_history = prox_hist
#prob.heur_history = heur_history
#prob.save_data()
#print(prob.heur_history.updated_cnt)
#print(prob.heur_history.leader_utility_history)
prob.heur_beta = 0.6
prob.heur_iterations()
#prob2 = Dispatching([prob.evs, prob.stations, prob.evs_loc, prob.sts_loc, prob.p_min, prob.p_max, prob.target_evs, prob.evs_load, prob.evs_priority, prob.max_elec, prob.max_evs])
#prob2.heur_iterations()