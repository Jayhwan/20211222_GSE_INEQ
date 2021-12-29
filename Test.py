from Prob_Multi_Hour_EV import *
import matplotlib.pyplot as plt

with open('grad.pkl', 'rb') as f:
    prob1 = pickle.load(f)

with open('prox.pkl', 'rb') as f:
    prob2 = pickle.load(f)


print("Gradient Utility Changes\n", prob1.leader_utility_history)
print("Proximal Utility Changes\n", prob2.leader_utility_history)

u1 = prob1.leader_utility_history
u2 = prob2.leader_utility_history
print(u1)
if len(u1) < len(u2):
    while len(u1) != len(u2):
        u1 += [u1[-1]]
else:
    while len(u1) != len(u2):
        u2 += [u2[-1]]

plt.figure()
plt.plot(u1[:40], color='r', label='GRAD')
plt.plot(u2[:40], color='b', label='FROX')
plt.xlabel("Iteration")
plt.ylabel("Leader Utility")
plt.legend()
plt.show()

