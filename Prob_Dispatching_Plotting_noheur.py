from Prob_Dispatching import *
import pandas as pd

with open("Dispatching/Dispatching_main.pkl", 'rb') as f:
    prob = pickle.load(f)
with open("Dispatching/Dispatching_main_grad_history.pkl", 'rb') as f:
    grad_history = pickle.load(f)
with open("Dispatching/Dispatching_main_heur_history_heur_history.pkl", 'rb') as f:
    heur_history = pickle.load(f)
with open("Dispatching/Dispatching_main_prox_history.pkl", 'rb') as f:
    prox_history = pickle.load(f)

prob.grad_history = grad_history
prob.prox_history = prox_history
prob.heur_history = heur_history

# 1-1 City map # 1-2 Target Load

# 1-2
"""
plt.figure()
plt.bar(np.arange(prob.evs), prob.evs_load, label="Require Load($E_{m}$)")
plt.ylim(0, 1.2)
plt.legend()
plt.xlabel("EVs", fontsize=12)
plt.ylabel("Electricity", fontsize=12)
plt.show()
"""


# 2-1 Stations target EV number 2-2 EVs' initial action

plt.figure()
width=0.5
x1 = np.array([1,3,5, 7, 9]) - width/2
x2 = np.array([1,3,5, 7, 9])
x3 = np.array([1,3,5, 7, 9]) + width/2

i=5
s = np.copy(prob.grad_history.followers_decision_history[-1][i])
for k in range(5):
    if s[k] <0.01:
        s[k]=0.01
plt.bar(x1,s, width=width, color='tab:red', label="Gradient Algorithm")

s = np.copy(prob.heur_history.followers_decision_history[-1][i])
for k in range(5):
    if s[k] <0.01:
        s[k]=0.01
#plt.bar(x2,s, width=width, color='tab:blue',label="Heuristic Algorithm")

s = np.copy(prob.prox_history.followers_decision_history[-1][i])
for k in range(5):
    if s[k] <0.01:
        s[k]=0.01
plt.bar(x3,s, width=width, color='tab:green', label="Proximal Algorithm")

ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=14, rotation=0)
plt.legend(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.tight_layout()
plt.show()


# 3-1 Leader's converging progress of each algorithm #3-2 EV's converging progress of each algorithm

iter = 40
plt.figure(figsize=(5,5))
#print(prob.grad_history.followers_utility_history)
gu = []
pu = []
for i in range(iter):
    gu += [np.sum(np.array(prob.grad_history.followers_utility_history[i]))]
    pu += [np.sum(np.array(prob.prox_history.followers_utility_history[i]))]
print(gu, pu)
#print(prob.prox_history.followers_utility_history)
plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.grad_history.leader_utility_history)))[:iter],color='tab:red', label='Gradient Algorithm')
#plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.heur_history.leader_utility_history)[:iter])),color='tab:blue', label='Heuristic Algorithm')
plt.plot(-np.array(prob.prox_history.leader_utility_history)[:iter],color='tab:green', label='Proximal Algorithm')
plt.legend(fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Leader Objective", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

g = np.zeros((prob.evs, iter))
h = np.zeros((prob.evs, iter))
p = np.zeros((prob.evs, iter))
avg_g = np.zeros((iter))
avg_h = np.zeros((iter))
avg_p = np.zeros((iter))
#ev = 20
for ev in range(prob.evs):
    for i in range(iter):
        g[ev][i] = prob.grad_history.followers_utility_history[i][ev]
        h[ev][i] = prob.heur_history.followers_utility_history[i][ev]
        p[ev][i] = prob.prox_history.followers_utility_history[i][ev]
avg_g = np.sum(g, axis=0)/prob.evs
avg_h = np.sum(h, axis=0)/prob.evs
avg_p = np.sum(p, axis=0)/prob.evs
plt.figure()
#for ev in range(prob.evs):
#    plt.plot(g[ev],color='tab:red', label='Gradient Algorithm')
#    plt.plot(h[ev],color='tab:blue', label='Heuristic Algorithm')
#    plt.plot(p[ev],color='tab:green', label='Proximal Algorithm')

plt.plot(avg_g,color='tab:red', ls='--', label='Gradient Algorithm')
#plt.plot(avg_h,color='tab:blue', ls='--', label='Average Heuristic Algorithm')
plt.plot(avg_p,color='tab:green', ls='--', label='Proximal Algorithm')
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Follower Objective", fontsize=14)
plt.show()

# 3-3

iter = 40
iter_h = iter
g = np.zeros((prob.evs, iter))
h = np.zeros((prob.evs, iter_h))
p = np.zeros((prob.evs, iter))
avg_g = np.zeros((iter))
avg_h = np.zeros((iter))
avg_p = np.zeros((iter))
#ev = 20
for ev in range(prob.evs):
    for i in range(iter):
        g[ev][i] = prob.grad_history.followers_utility_history[i][ev]
        p[ev][i] = prob.prox_history.followers_utility_history[i][ev]
    for i in range(iter_h):
        h[ev][i] = prob.heur_history.followers_utility_history[i][ev]
avg_g = np.sum(g, axis=0)/prob.evs
avg_h = np.sum(h, axis=0)/prob.evs
avg_p = np.sum(p, axis=0)/prob.evs

plt.figure(figsize=(5,5))
plt.plot(avg_g,color='tab:red', label='Gradient Algorithm')
plt.plot(avg_p,color='tab:green', label='Proximal Algorithm')
plt.legend(fontsize=14)
plt.ylabel("Averaged Follower Objective", fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(5,5))
grid = plt.GridSpec(2, 1, hspace=0, wspace=0)
ax = fig.add_subplot(grid[:,0])
ax1 = fig.add_subplot(grid[0,0])
#ax2 = fig.add_subplot(grid[1,1])
ax3 = fig.add_subplot(grid[1,0])
#ax4 = fig.add_subplot(grid[:,0])

for ev in range(prob.evs):
    #plt.plot(g[ev],color='tab:red', label='Gradient Algorithm')
    #plt.plot(h[ev],color='tab:blue', label='Heuristic Algorithm')
    if ev==0:
        ax1.plot(g[ev]-g[ev][-1], color='tab:red', ls='--', alpha=0.8, linewidth=0.6, label='Gradient Algorithm')
        #ax2.plot(h[ev]-h[ev][-1], color='tab:blue', ls='--', alpha=0.8, linewidth=0.6, label='Heuristic Algorithm')
        ax3.plot(p[ev]-p[ev][-1],color='tab:green', ls='--', alpha=0.8, linewidth=0.6,label='Proximal Algorithm')
    else:
        ax1.plot(g[ev] - g[ev][-1], color='tab:red', ls='--', linewidth=0.8,)
        #ax2.plot(h[ev] - h[ev][-1], color='tab:blue', ls='--', linewidth=0.8)
        ax3.plot(p[ev] - p[ev][-1], color='tab:green', ls='--', linewidth=0.8)

ax1.plot(np.zeros(iter), color='k', ls= '--', linewidth=1)
#ax2.plot(np.zeros(iter), color='k', ls='--', linewidth=1)
ax3.plot(np.zeros(iter), color='k', ls='--', linewidth=1)

#ax4.plot(avg_g,color='tab:red', ls='--', label='Gradient Algorithm')
#ax4.plot(avg_h,color='tab:blue', ls='--', label='Heuristic Algorithm')
#ax4.plot(avg_p,color='tab:green', ls='--', label='Proximal Algorithm')

#plt.plot(avg_g,color='tab:red', ls='--', label='Average Gradient Algorithm')
#plt.plot(avg_h,color='tab:blue', ls='--', label='Average Heuristic Algorithm')
#plt.plot(avg_g-avg_g[-1]+3.5,color='r', label='Average Proximal Algorithm')
#plt.plot(avg_h-avg_h[-1]+2.5,color='b', label='Average Proximal Algorithm')
#plt.plot(avg_p-avg_p[-1],color='g', label='Average Proximal Algorithm')
#plt.legend()
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.xlabel("Iteration", fontsize=14)
#plt.ylabel("Follower Objective", fontsize=14)
ax1.set_xlim([0,iter])
ax1.set_xticks([])
#ax2.set_xlim([0,iter])
#ax2.set_xticks([])
ax3.set_xlim([0,iter])
ax1.set_ylim([-1.5, 1.5])
#ax2.set_ylim([-2.5, 2.5])
ax3.set_ylim([-1.5, 1.5])
#ax4.set_ylim([-2.5, 2.5])
ax1.legend(fontsize=14)
#ax2.legend(fontsize=14)
ax3.legend(fontsize=14)
ax3.set_xticks([10, 20, 30, 40])

#ax.spines['top'].set_color('none')
#ax.spines['bottom'].set_color('none')
#ax.spines['left'].set_color('none')
#ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax.set_ylabel("Difference", fontsize=14)
ax.set_xlabel("Iteration", fontsize=14)

#ax4.legend(fontsize=14)
#ax4.set_ylabel("Averaged Follower Objective", fontsize=14)
#ax4.set_xlabel("Iteration", fontsize=14)

grid.tight_layout(fig)
plt.show()


# 4-1 Converged pricing of Each algorithm(bar chart) 4-2 Converged action of EVs (bar chart), some EVs

plt.figure()
width=0.5
x1 = np.array([1,3,5, 7, 9]) - width/2
#x2 = np.array([1,3,5, 7, 9])
x3 = np.array([1,3,5, 7, 9]) + width/2
x = [x1, x3] #, x2, x3]

print(prob.grad_history.leader_decision_history[-1], prob.grad_history.leader_utility_history[-1])
print(prob.heur_history.leader_decision_history[-1], prob.heur_history.leader_utility_history[-1])
plt.bar(x1, prob.grad_history.leader_decision_history[-1],color='tab:red', width=width, label="Gradient Algorithm")
#plt.bar(x2, prob.heur_history.leader_decision_history[-1],color='tab:blue', width=width, label="Heuristic Algorithm")
plt.bar(x3, prob.prox_history.leader_decision_history[-1],color='tab:green', width=width, label="Proximal Algorithm")

ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Electricity Price", fontsize=14)
plt.tight_layout()
plt.show()


plt.figure()
width=0.5
x1 = np.array([1,3,5, 7, 9]) - width
x2 = np.array([1,3,5, 7, 9])
x3 = np.array([1,3,5, 7, 9]) + width
x = [x1, x2, x3]
j=0

for i in [5, 14, 22]:
    print(x[j])
    s = np.copy(prob.grad_history.followers_decision_history[-1][i])
    for k in range(5):
        if s[k] <0.01:
            s[k]=0.01
    plt.bar(x[j], s, width=width, label="EV"+str(i+1))
    j += 1
ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=12, rotation=0)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel("Stations", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.show()


# 5-1 EVs in each station compare with target EV number (bar chart)
plt.figure(figsize=(12, 6))
width=0.25
x1 = np.array([1,3,5,7,9]) - width#[1-width/2,3-width/2,5-width/2,7-width/2,9-width/2]
x2 = np.array([1,3,5,7,9])
#x3 = np.array([1,3,5,7,9]) + width/2
x4 = np.array([1,3,5,7,9]) + width
plt.bar(x1, prob.target_evs, color='dimgrey', width=width, label="Target($V^{m}$)")
plt.bar(x2, np.sum(prob.grad_history.followers_decision_history[-1], axis=0),color='tab:red', width=width, label="Gradient Algorithm")
#plt.bar(x3, np.sum(prob.heur_history.followers_decision_history[-1], axis=0),color='tab:blue', width=width, label="Heuristic Algorithm")
plt.bar(x4, np.sum(prob.prox_history.followers_decision_history[-1], axis=0),color='tab:green', width=width, label="Proximal Algorithm")
plt.legend(fontsize=14)
ticklabel=['1', '2', '3', '4', '5']
plt.xticks(np.array([1,3,5,7,9]), ticklabel, fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Number of EVs", fontsize=14)
plt.tight_layout()
plt.show()
