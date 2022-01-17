from Prob_ESS import *

#[active_users, passive_users, times, alpha, beta_s, beta_b, q_max, q_init, c_max, c_min, grid_price, tax, soh, active_require_loads, passive_require_loads]
#params_5_6 = [5, 1, 6, 0.9956, 0.99, 1.01, 100, 0, 10, 10, 1, 0.00001, 0.00001, np.kron(np.ones((5, 1)), np.array([0.3, 1, 0.3, 0.1, 4, 1])), 0.1 * np.random.random((1, 6))]
#prob = EET_Game(params_5_6, "ess_grad.pkl")
#prob.prox_iterations()

for n in range(1,60):#[30]:#[1,3,5,7,9,11,13,15,17,19, 20, 30, 40, 50, 60, 80, 100]:
    print("User number :", n)
    #n = 30
    t = 12
    x = np.load("load_123.npy", allow_pickle=True)
    params_5_6 = [n, 100-n, t, 0.9956, 0.99, 1.01, 1000, 0, 1000, 1000, 1, 0.00001, 0.001, x[:n, :t], x[n:100, :t]]
    prob = EET_Game(params_5_6, "results/prox/"+str(n)+"_"+str(t)+".pkl")
    prob.approx_ve_eps = 1e-5
    prob.ve_step_size = 0.001
    #with open("results/grad/"+str(n)+"_"+str(t)+".pkl", 'rb') as f:
    #    prob = pickle.load(f)
    prob.prox_iterations()