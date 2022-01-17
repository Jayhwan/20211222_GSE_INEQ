from Prob_ESS import *

#[active_users, passive_users, times, alpha, beta_s, beta_b, q_max, q_init, c_max, c_min, grid_price, tax, soh, active_require_loads, passive_require_loads]
#x = 3*(np.random.random((99, 24))-0.4)
#np.save("tmp.npy", x)
#n = 60
#t = 12
#x = np.load("load_123.npy", allow_pickle=True)
#params_5_6 = [n, 100-n, t, 0.9956, 0.99, 1.01, 1000, 0, 1000, 1000, 1, 0.00001, 0.001, x[:n, :t], x[n:100, :t]]
#prob = EET_Game(params_5_6, "results/grad/"+str(n)+"_"+str(t)+".pkl")
#with open("results/grad/"+str(n)+"_"+str(t)+".pkl", 'rb') as f:
#    prob = pickle.load(f)

#prob.approx_ve_eps = 1e-5
#prob.active_epsilon = 1e-4
#print(prob.par_history)
#prob.iterations()
user = []
prox_user = []
ec_list = []
par_list = []
prox_ec_list = []
prox_par_list = []
for n in [1,3,5, 10, 20, 30, 40]:#[1,3,5,20, 30, 40, 50, 60, 80, 100]:
    #n = 30
    t = 12
    with open("results/grad/" + str(n) + "_" + str(t) + ".pkl", 'rb') as f:
        prob = pickle.load(f)
    with open("results/prox/" + str(n) + "_" + str(t) + ".pkl", 'rb') as f:
        prox_prob = pickle.load(f)
    if len(prob.leader_utility_history) < 0:
        continue
    else:
        user += [n]
        ec_list += [prob.leader_utility()]
        par_list += [prob.par()]
        prox_ec_list += [prox_prob.leader_utility()]
        prox_par_list += [prox_prob.par()]
    print(prob.par_history[-10:])
    plt.subplot(2, 1, 1)
    plt.title("[Convergence] Active user :"+str(n))
    print("User ", str(n), prob.leader_utility_history)
    print("User ", str(n), prox_prob.leader_utility_history)
    plt.plot(-np.array(prob.leader_utility_history), color='r', label="Energy Cost(Gradient Descent)")
    plt.plot(-prox_prob.leader_utility_history[-1]*np.ones(len(prob.leader_utility_history)), linestyle='--', color='k', label="Energy cost(Proximal Algorithm)")
    plt.legend()
    #plt.xlabel("Iteration")
    plt.xticks(visible=False)
    plt.subplot(2, 1, 2)
    #plt.title("user :"+str(n))
    plt.plot(prob.par_history, color='r', label = "PAR(Gradient Descent)")
    plt.plot(prox_prob.par_history[-1]*np.ones(len(prob.par_history)), linestyle='--', color='k', label="PAR(Proximal Algorithm)")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()

plt.subplot(2, 1, 1)
plt.title("[Performance]")
plt.plot(user, -np.array(ec_list), color='r', label="Energy cost(Gradient Descent)")
plt.plot(user, -np.array(prox_ec_list), color='k', label="Energy cost(Proximal Algorithm)")
plt.legend()
#plt.xlabel("Iteration")
plt.xticks(visible=False)
plt.subplot(2, 1, 2)
#plt.title("user :"+str(n))
plt.plot(user, par_list, color='r', label = "PAR(Gradient Descent)")
plt.plot(user, prox_par_list, color='k', label = "PAR(Proximal Algorithm)")
plt.xlabel("Number of Active Users")
plt.legend()
plt.show()