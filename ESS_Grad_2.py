from Prob_ESS import *

#[active_users, passive_users, times, alpha, beta_s, beta_b, q_max, q_init, c_max, c_min, grid_price, tax, soh, active_require_loads, passive_require_loads]
#x = 3*(np.random.random((99, 24))-0.4)
#np.save("tmp.npy", x)
for n in range(1,60):
    print("User number :", n)
    #n = 30
    t = 12
    x = np.load("load_123.npy", allow_pickle=True)
    params_5_6 = [n, 100-n, t, 0.9956, 0.99, 1.01, 1000, 0, 1000, 1000, 1, 0.00001, 0.001, x[:n, :t], x[n:100, :t]]
    prob = EET_Game(params_5_6, "results/grad/"+str(n)+"_"+str(t)+".pkl")
    prob.approx_ve_eps = 1e-5
    prob.ve_step_size = 0.001
    prob.max_iter = 50
    #with open("results/grad/"+str(n)+"_"+str(t)+".pkl", 'rb') as f:
    #    prob = pickle.load(f)
    prob.iterations()
#n = 30
#t = 12
#x = np.load("load_123.npy", allow_pickle=True)
#params_5_6 = [n, 100-n, t, 0.9956, 0.99, 1.01, 1000, 0, 1000, 1000, 1, 0.00001, 0.001, x[:n, :t], x[n:100, :t]]
#prob = EET_Game(params_5_6, "results/grad/"+str(n)+"_"+str(t)+".pkl")
#with open("results/grad/"+str(n)+"_"+str(t)+".pkl", 'rb') as f:
#    prob = pickle.load(f)
#prob.iterations()
#print(prob.par_history)
#prob.print_information()
"""
plt.figure()
plt.plot(-np.array(prob.leader_utility_history), label="Leader cost")
plt.xlabel("Iteration")
plt.legend()
plt.show()
plt.figure()
plt.plot(prob.par_history, label = "par")
plt.xlabel("Iteration")
plt.legend()
plt.show()"""