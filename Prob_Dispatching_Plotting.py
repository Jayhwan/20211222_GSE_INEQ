from Prob_Dispatching import *

with open("Dispatching/Dispatching_main.pkl", 'rb') as f:
    prob = pickle.load(f)
with open("Dispatching/Dispatching_main_grad_history.pkl", 'rb') as f:
    grad_history = pickle.load(f)
with open("Dispatching/Dispatching_main_heur_history.pkl", 'rb') as f:
    heur_history = pickle.load(f)
with open("Dispatching/Dispatching_main_prox_history.pkl", 'rb') as f:
    prox_history = pickle.load(f)

prob.grad_history = grad_history
prob.prox_history = prox_history
prob.heur_history = heur_history

# 1-1 City map # 1-2 Target Load

marker_size = 8
ev_x = []
ev_y = []
st_x = []
st_y = []
for ev_loc in prob.evs_loc:
    ev_x += [ev_loc[0]]
    ev_y += [ev_loc[1]]
    # plt.plot(ev_loc, marker='o', marker_size=marker_size, color = 'r')
coordinate = []
i=0
for st_loc in prob.sts_loc:
    i += 1
    coordinate += [('Station '+str(i)+'('+str(prob.target_evs[i-1])+')')]
    st_x += [st_loc[0]]
    st_y += [st_loc[1]]
plt.figure(figsize=(10, 6))
        # plt.plot(st_loc, marker='o', marker_size=marker_size, color = 'k')
plt.scatter(st_x, st_y, marker='s', s=marker_size+5, color='k', label='Stations')
plt.scatter(ev_x, ev_y, s=marker_size+5, color='r', label='EVs')

for i in range(prob.evs):
    if i in [5]:#, 14, 22]:
        plt.text(ev_x[i]-.15, ev_y[i]-.30, "EV"+str(i+1), color='r', fontsize=14)

for i in range(prob.stations):
    plt.text(st_x[i]-.28, st_y[i]-.28, "Station"+str(i+1)+" ("+str(prob.target_evs[i])+")", color='k', fontsize=14)
plt.xlim(0, 5.5)
plt.ylim(0, 6)
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right',ncol=2, fontsize=16)
plt.show()

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
x1 = np.array([1,3,5,7,9]) - width/2#[1-width/2,3-width/2,5-width/2,7-width/2,9-width/2]
x2 = np.array([1,3,5,7,9]) + width/2
plt.bar(x1, prob.target_evs, color='r', width=width, label="Target($V^{m}$)")
plt.bar(x2, np.sum(prob.grad_history.followers_decision_history[0], axis=0), color='k', width=width, label="Identical Price")
plt.legend(fontsize=12)
ticklabel=['1', '2', '3', '4', '5']
plt.xticks(np.array([1,3,5,7,9]), ticklabel, fontsize=12, rotation=0)
plt.xlabel("Stations", fontsize=12)
plt.ylabel("Number of EVs", fontsize=12)
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
    s = np.copy(prob.grad_history.followers_decision_history[0][i])
    for k in range(5):
        if s[k] <0.01:
            s[k]=0.01
    plt.bar(x[j], s, width=width, label="EV"+str(i+1))
    j += 1
ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=12, rotation=0)
plt.legend()
plt.xlabel("Stations", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.show()

plt.figure()
width=0.5
x1 = np.array([1,3,5, 7, 9]) - width
x2 = np.array([1,3,5, 7, 9])
x3 = np.array([1,3,5, 7, 9]) + width

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
plt.bar(x2,s, width=width, color='tab:blue',label="Heuristic Algorithm")

s = np.copy(prob.prox_history.followers_decision_history[-1][i])
for k in range(5):
    if s[k] <0.01:
        s[k]=0.01
plt.bar(x3,s, width=width, color='tab:green', label="Proximal Algorithm")

ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=14, rotation=0)
plt.legend()
plt.yticks(fontsize=14)
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.show()


# 3-1 Leader's converging progress of each algorithm #3-2 EV's converging progress of each algorithm

plt.figure()
plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.grad_history.leader_utility_history)[:80])),color='tab:red', label='Gradient Algorithm')
plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.heur_history.leader_utility_history)[:80])),color='tab:blue', label='Heuristic Algorithm')
plt.plot(-np.array(prob.prox_history.leader_utility_history)[:80],color='tab:green', label='Proximal Algorithm')
plt.legend()
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Leader Objective", fontsize=14)
plt.show()

g = []
h = []
p = []
ev = 20
for i in range(80):
    g += [prob.grad_history.followers_utility_history[i][ev]]
    h += [prob.heur_history.followers_utility_history[i][ev]]
    p += [prob.prox_history.followers_utility_history[i][ev]]

plt.figure()
plt.plot(g,color='tab:red', label='Gradient Algorithm')
plt.plot(h,color='tab:blue', label='Heuristic Algorithm')
plt.plot(p,color='tab:green', label='Proximal Algorithm')
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Follower Objective", fontsize=14)
plt.show()

# 4-1 Converged pricing of Each algorithm(bar chart) 4-2 Converged action of EVs (bar chart), some EVs

plt.figure()
width=0.5
x1 = np.array([1,3,5, 7, 9]) - width
x2 = np.array([1,3,5, 7, 9])
x3 = np.array([1,3,5, 7, 9]) + width
x = [x1, x2, x3]

print(prob.grad_history.leader_decision_history[-1], prob.grad_history.leader_utility_history[-1])
print(prob.heur_history.leader_decision_history[-1], prob.heur_history.leader_utility_history[-1])
plt.bar(x1, prob.grad_history.leader_decision_history[-1],color='tab:red', width=width, label="Gradient Algorithm")
plt.bar(x2, prob.heur_history.leader_decision_history[-1],color='tab:blue', width=width, label="Heuristic Algorithm")
plt.bar(x3, prob.prox_history.leader_decision_history[-1],color='tab:green', width=width, label="Proximal Algorithm")

ticklabel=['1', '2', '3', '4', '5']
plt.xticks(x2, ticklabel, fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.legend()
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Electricity Price", fontsize=14)
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
plt.legend()
plt.xlabel("Stations", fontsize=12)
plt.ylabel("Probability", fontsize=12)
plt.show()


# 5-1 EVs in each station compare with target EV number (bar chart)
plt.figure(figsize=(12, 6))
width=0.25
x1 = np.array([1,3,5,7,9]) - 3*width/2#[1-width/2,3-width/2,5-width/2,7-width/2,9-width/2]
x2 = np.array([1,3,5,7,9]) - width/2
x3 = np.array([1,3,5,7,9]) + width/2
x4 = np.array([1,3,5,7,9]) + 3*width/2
plt.bar(x1, prob.target_evs, color='dimgrey', width=width, label="Target($V^{m}$)")
plt.bar(x2, np.sum(prob.grad_history.followers_decision_history[-1], axis=0),color='tab:red', width=width, label="Gradient Algorithm")
plt.bar(x3, np.sum(prob.heur_history.followers_decision_history[-1], axis=0),color='tab:blue', width=width, label="Heuristic Algorithm")
plt.bar(x4, np.sum(prob.prox_history.followers_decision_history[-1], axis=0),color='tab:green', width=width, label="Proximal Algorithm")
plt.legend(fontsize=12)
ticklabel=['1', '2', '3', '4', '5']
plt.xticks(np.array([1,3,5,7,9]), ticklabel, fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.xlabel("Stations", fontsize=14)
plt.ylabel("Number of EVs", fontsize=14)
plt.show()
