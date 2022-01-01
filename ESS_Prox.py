from Prob_ESS import *

#[active_users, passive_users, times, alpha, beta_s, beta_b, q_max, q_init, c_max, c_min, grid_price, tax, soh, active_require_loads, passive_require_loads]
params_5_6 = [5, 1, 6, 0.9956, 0.99, 1.01, 100, 0, 10, 10, 1, 0.00001, 0.00001, np.kron(np.ones((5, 1)), np.array([0.3, 1, 0.3, 0.1, 4, 1])), 0.1 * np.random.random((1, 6))]
prob = EET_Game(params_5_6, "ess_grad.pkl")
prob.prox_iterations()
