from Prob_Dispatching import *

#with open("Dispatching/파라미터저장용.pkl", 'rb') as f:
#    prob_ = pickle.load(f)
#evs_load = prob_.evs_load
with open("Dispatching/Dispatching_main.pkl", 'rb') as f:
    prob = pickle.load(f)

with open("Dispatching/Dispatching_main_grad_history.pkl", 'rb') as f:
    grad_history = pickle.load(f)
with open("Dispatching/Dispatching_main_heur_history.pkl", 'rb') as f:
    heur_history = pickle.load(f)
with open("Dispatching/Dispatching_main_prox_history.pkl", 'rb') as f:
    prox_history = pickle.load(f)
#prob.change_filename("Dispatching/Dispatching_main.pkl")

#prob.evs_load = evs_load
print(prob.evs, prob.stations)
print(prob.evs_loc, prob.sts_loc)
print(prob.p_min, prob.p_max)
print(prob.target_evs)
print(prob.evs_load)
print(prob.evs_priority)
print(prob.max_elec)
print(prob.max_evs)
prob.grad_history = grad_history
prob.prox_history = prox_history
prob.heur_history = heur_history
#prob.save_data()
#print(prob.heur_history.updated_cnt)
#print(prob.heur_history.leader_utility_history)
print(prob.heur_iterations())
#prob2 = Dispatching([prob.evs, prob.stations, prob.evs_loc, prob.sts_loc, prob.p_min, prob.p_max, prob.target_evs, prob.evs_load, prob.evs_priority, prob.max_elec, prob.max_evs])
#prob2.heur_iterations()