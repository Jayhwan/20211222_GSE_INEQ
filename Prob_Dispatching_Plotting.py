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

a = np.multiply(prob.grad_history.leader_decision_history[-1], np.sum(prob.grad_history.followers_decision_history[-1], axis=0))
b = np.multiply(prob.heur_history.leader_decision_history[-1], np.sum(prob.heur_history.followers_decision_history[-1], axis=0))
c = np.multiply(prob.prox_history.leader_decision_history[-1], np.sum(prob.prox_history.followers_decision_history[-1], axis=0))

print(a, b, c)
print(np.sum(a), np.sum(b), np.sum(c))
marker_size = 8
ev_x = []
ev_y = []
st_x = []
st_y = []
color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
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
plt.scatter(st_x, st_y, marker='s', s=marker_size+120, color=color_map, label='Station')

prob_ = prob.prox_history
ev_dest = np.argmax(prob_.followers_decision_history[-1], axis=-1)
print(prob_.followers_decision_history[-1][1])
ev_color = np.take(color_map, ev_dest)
plt.scatter(ev_x, ev_y, s=marker_size+20, color=ev_color, alpha=0.7, label='EV')

for i in range(prob.evs):
    plt.plot([ev_x[i], st_x[ev_dest[i]]], [ev_y[i], st_y[ev_dest[i]]], ls='--', alpha=0.3, color=ev_color[i])

for i in range(prob.evs):
    if i in [5]:#, 14, 22]:
        plt.text(ev_x[i]-.15, ev_y[i]-.30, "EV"+str(i+1-5), color='k', fontsize=12)

for i in range(prob.stations):
    plt.text(st_x[i]-.24, st_y[i]-.28, "Station"+str(i+1), color='k', fontsize=12)
plt.xlim(0, 5.5)
plt.ylim(0, 6)
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right',ncol=2, fontsize=16)

plt.show()

# 1-2 Include Table
fig = plt.figure(figsize=(9, 10))
grid = plt.GridSpec(4, 1, hspace=0)
ax1 = fig.add_subplot(grid[:-1,0])
ax1.scatter(st_x, st_y, marker='s', s=marker_size+120, color=color_map, label='Station')

ev_dest = np.argmax(prob_.followers_decision_history[-1], axis=-1)
print(prob_.followers_decision_history[-1][1])
ev_color = np.take(color_map, ev_dest)
ax1.scatter(ev_x, ev_y, s=marker_size+20, color=ev_color, alpha=0.7, label='EV')

for i in range(prob.evs):
    ax1.plot([ev_x[i], st_x[ev_dest[i]]], [ev_y[i], st_y[ev_dest[i]]], ls='--', alpha=0.3, color=ev_color[i])

for i in range(prob.evs):
    if i in [5]:#, 14, 22]:
        ax1.text(ev_x[i]-.15, ev_y[i]-.30, "EV"+str(i+1-5), color='k', fontsize=12)

for i in range(prob.stations):
    ax1.text(st_x[i]-.24, st_y[i]-.28, "Station"+str(i+1), color='k', fontsize=12)
ax1.set_xlim(0, 5.5)
ax1.set_ylim(0, 6)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend(loc='upper right',ncol=2, fontsize=16)

x = np.zeros((5,4))
x[:,0] = np.array([1,2,3,4,5])
x[:,1] = np.round(prob_.leader_decision_history[-1]-0.5, 1)
x[:,2] = prob.target_evs
x[:,3] = np.round(np.sum(prob_.followers_decision_history[-1], axis=0), 1)
print(x)
df = pd.DataFrame(x, columns=['Station', 'Electricity Price', 'Target EV($V^m$)', 'Expected EV($v^m$)'])
df['Station'] = df['Station'].apply(int)
df['Target EV($V^m$)'] = df['Target EV($V^m$)'].apply(int)
print(df)
a = df.values.tolist()
#a[0][0] = int(a[0][0])
for i in [0, 2]:
    for j in range(5):
        a[j][i] = int(a[j][i])
ax2 = fig.add_subplot(grid[-1,0])
table = ax2.table(cellText=a, colLabels=df.columns, loc='upper center', cellLoc = 'center', colWidths = [.1, .1, .1, .1], fontsize=100)
table.auto_set_font_size(False)
table.scale(2.5,1.5)
table.set_fontsize(14)

#table.scale(1,2)
ax2.axis('tight')
ax2.axis('off')
plt.tight_layout()
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
plt.tight_layout()
plt.show()


# 3-1 Leader's converging progress of each algorithm #3-2 EV's converging progress of each algorithm

iter = 40
plt.figure()
plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.grad_history.leader_utility_history)[:iter])),color='tab:red', label='Gradient Algorithm')
plt.plot(-np.hstack((prob.prox_history.leader_utility_history[0],np.array(prob.heur_history.leader_utility_history)[:iter])),color='tab:blue', label='Heuristic Algorithm')
plt.plot(-np.array(prob.prox_history.leader_utility_history)[:iter],color='tab:green', label='Proximal Algorithm')
plt.legend()
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Leader Objective", fontsize=14)
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

plt.plot(avg_g,color='tab:red', ls='--', label='Average Gradient Algorithm')
plt.plot(avg_h,color='tab:blue', ls='--', label='Average Heuristic Algorithm')
plt.plot(avg_p,color='tab:green', ls='--', label='Average Proximal Algorithm')
plt.legend()
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

fig = plt.figure(figsize=(8,6))
grid = plt.GridSpec(3, 2, hspace=0, wspace=0.05)
ax = fig.add_subplot(grid[:,1])
ax1 = fig.add_subplot(grid[0,1])
ax2 = fig.add_subplot(grid[1,1])
ax3 = fig.add_subplot(grid[2,1])
ax4 = fig.add_subplot(grid[:,0])

for ev in range(prob.evs):
    #plt.plot(g[ev],color='tab:red', label='Gradient Algorithm')
    #plt.plot(h[ev],color='tab:blue', label='Heuristic Algorithm')
    if ev==0:
        ax1.plot(g[ev]-g[ev][-1], color='tab:red', ls='--', alpha=0.8, linewidth=0.6, label='Gradient Algorithm')
        ax2.plot(h[ev]-h[ev][-1], color='tab:blue', ls='--', alpha=0.8, linewidth=0.6, label='Heuristic Algorithm')
        ax3.plot(p[ev]-p[ev][-1],color='tab:green', ls='--', alpha=0.8, linewidth=0.6,label='Proximal Algorithm')
    else:
        ax1.plot(g[ev] - g[ev][-1], color='tab:red', ls='--', linewidth=0.8,)
        ax2.plot(h[ev] - h[ev][-1], color='tab:blue', ls='--', linewidth=0.8)
        ax3.plot(p[ev] - p[ev][-1], color='tab:green', ls='--', linewidth=0.8)

ax1.plot(np.zeros(iter), color='k', ls= '--', linewidth=1)
ax2.plot(np.zeros(iter), color='k', ls='--', linewidth=1)
ax3.plot(np.zeros(iter), color='k', ls='--', linewidth=1)

ax4.plot(avg_g,color='tab:red', ls='--', label='Gradient Algorithm')
ax4.plot(avg_h,color='tab:blue', ls='--', label='Heuristic Algorithm')
ax4.plot(avg_p,color='tab:green', ls='--', label='Proximal Algorithm')

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
ax2.set_xlim([0,iter])
ax2.set_xticks([])
ax3.set_xlim([0,iter])
ax1.set_ylim([-1.5, 1.5])
ax2.set_ylim([-2.5, 2.5])
ax3.set_ylim([-1.5, 1.5])
#ax4.set_ylim([-2.5, 2.5])
ax1.legend(fontsize=14)
ax2.legend(fontsize=14)
ax3.legend(fontsize=14)
ax3.set_xticks([10, 20, 30, 40])

#ax.spines['top'].set_color('none')
#ax.spines['bottom'].set_color('none')
#ax.spines['left'].set_color('none')
#ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax.set_ylabel("Difference", fontsize=14)
ax.set_xlabel("Iteration", fontsize=14)

ax4.legend(fontsize=14)
ax4.set_ylabel("Averaged Follower Objective", fontsize=14)
ax4.set_xlabel("Iteration", fontsize=14)

grid.tight_layout(fig)
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
plt.tight_layout()
plt.show()
