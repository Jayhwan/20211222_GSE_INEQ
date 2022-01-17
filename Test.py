from Prob_Multi_Hour_EV import *
import matplotlib.pyplot as plt

with open('MultiHOurEV/grad.pkl', 'rb') as f:
    prob1 = pickle.load(f)

with open('MultiHOurEV/prox.pkl', 'rb') as f:
    prob2 = pickle.load(f)


print(len(prob1.followers_decision_history))
print(len(prob1.leader_decision_history))
print(len(prob1.leader_utility_history))

a = np.zeros(3)
with open('results/haha.pkl', "wb") as f:
    pickle.dump(a, f)

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

mark_size=0
plt.figure()
plt.plot(u1[:100], label="Gradient Algorithm", color='r', marker='.', markersize=mark_size)
plt.plot(u2[:100], label="Proximal Algorithm", marker='.', markersize=mark_size)
plt.plot(118*np.ones(100), label="Heuristic Pricing", linestyle="--", markersize=mark_size)
#plt.plot(u1[:50], color='r',marker='.', markersize=mark_size)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Leader Objective", fontsize=12)
plt.ylim(top=200)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()

prob1_follower = MultiHourEV(params_24_50)
prob2_follower = MultiHourEV(params_24_50)

fu1 = []
fu2 = []
f1a = []
f2a = []
print(len(prob1.followers_decision_history))
print(len(prob1.leader_decision_history))
print(len(prob1.leader_utility_history))
print(np.array(prob1.followers_decision_history).shape)
for i in range(100):
    if i< len(prob1.leader_decision_history):
        prob1_follower.leader.decision = prob1.leader_decision_history[i]
    prob2_follower.leader.decision = prob2.leader_decision_history[i]
    for j in range(len(prob1.followers)):
        #print(i)
        if i< len(prob1.followers_decision_history):
            prob1_follower.followers[j].decision = prob1.followers_decision_history[i][j]
        prob2_follower.followers[j].decision = prob2.followers_decision_history[i][j]
    f1a += [prob1_follower.followers[10].decision]
    f2a += [prob2_follower.followers[10].decision]
    fu1 += [prob1_follower.followers_utility()[0]]
    fu2 += [prob2_follower.followers_utility()[0]]
mark_size = 1
print(fu1, fu2)
plt.figure()
plt.plot(fu1, label="Gradient Algorithm", color='r')#, marker='.')#, markersize=mark_size)
plt.plot(fu2, label="Proximal Algorithm")#, marker='.')#, markersize=mark_size)
plt.plot(24.3*np.ones(100), label="Heuristic Pricing", linestyle="--")#, markersize=mark_size)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Follower Objective", fontsize=12)
#plt.ylim(top=200)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()

plt.figure()
plt.plot(f1a[9], label="Gradient Algorithm", color='r')
plt.plot(f2a[50], label="Proximal Algorithm")
plt.plot(f1a[2], label="Heuristic Pricing")
plt.xlabel("Hours(h)")
plt.ylabel("Load(kWh)")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()