from Prob_Dispatching import *
import pandas as pd
import matplotlib.pyplot as plt
grad = []
prox = []

# 5, 25
exp_num = list(range(35))
for i in [2,3,4,6,28]:
    exp_num.pop(i)

# 5, 50

ev_ = 50
sts_ = 5
exp_num = list(range(1, 15))
for i in [4]:
    exp_num.pop(i)

for exp in exp_num:
    with open("results/average/"+str(ev_)+"ev_"+str(sts_)+"st"+"/env_"+str(exp)+".pkl", 'rb') as f:
        prob = pickle.load(f)
    with open("results/average/"+str(ev_)+"ev_"+str(sts_)+"st"+"/env_"+str(exp)+"_grad_history.pkl", 'rb') as f:
        grad_history = pickle.load(f)
    with open("results/average/"+str(ev_)+"ev_"+str(sts_)+"st"+"/env_"+str(exp)+"_heur_history.pkl", 'rb') as f:
        heur_history = pickle.load(f)
    with open("results/average/"+str(ev_)+"ev_"+str(sts_)+"st"+"/env_"+str(exp)+"_prox_history.pkl", 'rb') as f:
        prox_history = pickle.load(f)

    prob.grad_history = grad_history
    prob.prox_history = prox_history
    prob.heur_history = heur_history

    grad += [grad_history.leader_utility_history[-1]]
    prox += [prox_history.leader_utility_history[-1]]
    """
    #print(grad_history.leader_utility_history)
    #print(heur_history.leader_utility_history)


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
    i = 0
    for st_loc in prob.sts_loc:
        i += 1
        coordinate += [('Station ' + str(i) + '(' + str(prob.target_evs[i - 1]) + ')')]
        st_x += [st_loc[0]]
        st_y += [st_loc[1]]
    #plt.figure(figsize=(10, 6))
    # plt.plot(st_loc, marker='o', marker_size=marker_size, color = 'k')
    #plt.scatter(st_x, st_y, marker='s', s=marker_size + 120, color=color_map, label='Station')

    
    prob_ = prob.grad_history
    
    # 1-2 Include Table
    fig = plt.figure(figsize=(9, 10))
    grid = plt.GridSpec(4, 1, hspace=0)
    ax1 = fig.add_subplot(grid[:-1, 0])
    ax1.scatter(st_x, st_y, marker='s', s=marker_size + 120, color=color_map, label='Station')

    mi = -1#np.argmax(prob.grad_history.leader_utility_history)
    print(prob_.leader_utility_history)
    ev_dest = np.argmax(prob_.followers_decision_history[mi], axis=-1)
    #print(prob_.followers_decision_history[-1][1])
    ev_color = np.take(color_map, ev_dest)
    ax1.scatter(ev_x, ev_y, s=marker_size + 20, color=ev_color, alpha=0.7, label='EV')

    for i in range(prob.evs):
        ax1.plot([ev_x[i], st_x[ev_dest[i]]], [ev_y[i], st_y[ev_dest[i]]], ls='--', alpha=0.3, color=ev_color[i])

    for i in range(prob.evs):
        if i in []:#5]:  # , 14, 22]:
            ax1.text(ev_x[i] - .15, ev_y[i] - .30, "EV" + str(i + 1 - 5), color='k', fontsize=12)

    for i in range(prob.stations):
        ax1.text(st_x[i] - .24, st_y[i] - .28, "Station" + str(i + 1), color='k', fontsize=12)

    #ax1.text(0.5, 5.8, "Experiment "+str(exp+1)+" Gradient Algorithm", color='k', fontsize=12)
    #ax1.text(0.5, 5.5, "Leader Utility : "+str(prob.grad_history.leader_utility_history[mi]), color='k', fontsize=12)
    ax1.set_xlim(0, 5.5)
    ax1.set_ylim(0, 6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='upper right', ncol=2, fontsize=16)

    x = np.zeros((5, 4))
    x[:, 0] = np.array([1, 2, 3, 4, 5])
    x[:, 1] = np.round(prob_.leader_decision_history[mi] - 0.5, 1)
    x[:, 2] = prob.target_evs
    x[:, 3] = np.round(np.sum(prob_.followers_decision_history[mi], axis=0), 1)
    print(x)
    df = pd.DataFrame(x, columns=['Station', 'Electricity Price', 'Target EV($V^m$)', 'Expected EV($v^m$)'])
    df['Station'] = df['Station'].apply(int)
    #df['Target EV($V^m$)'] = df['Target EV($V^m$)'].apply(int)
    print(df)
    a = df.values.tolist()
    # a[0][0] = int(a[0][0])
    for i in [0, 2]:
        for j in range(5):
            if i == 0:
                a[j][i] = int(a[j][i])
            if i != 0:
                a[j][i] = np.round(a[j][i], 1)
    ax2 = fig.add_subplot(grid[-1, 0])
    table = ax2.table(cellText=a, colLabels=df.columns, loc='upper center', cellLoc='center',
                      colWidths=[.1, .1, .1, .1], fontsize=100)
    table.auto_set_font_size(False)
    table.scale(2.5, 1.5)
    table.set_fontsize(14)

    # table.scale(1,2)
    ax2.axis('tight')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    prob_ = prob.prox_history

    # 1-2 Include Table
    fig = plt.figure(figsize=(9, 10))
    grid = plt.GridSpec(4, 1, hspace=0)
    ax1 = fig.add_subplot(grid[:-1, 0])
    ax1.scatter(st_x, st_y, marker='s', s=marker_size + 120, color=color_map, label='Station')


    mi = -1 # np.argmax(prob_.leader_utility_history)

    ev_dest = np.argmax(prob_.followers_decision_history[mi], axis=-1)
    #print(prob_.followers_decision_history[-1][1])
    ev_color = np.take(color_map, ev_dest)
    ax1.scatter(ev_x, ev_y, s=marker_size + 20, color=ev_color, alpha=0.7, label='EV')

    for i in range(prob.evs):
        ax1.plot([ev_x[i], st_x[ev_dest[i]]], [ev_y[i], st_y[ev_dest[i]]], ls='--', alpha=0.3, color=ev_color[i])

    for i in range(prob.evs):
        if i in []:#5]:  # , 14, 22]:
            ax1.text(ev_x[i] - .15, ev_y[i] - .30, "EV" + str(i + 1 - 5), color='k', fontsize=12)

    for i in range(prob.stations):
        ax1.text(st_x[i] - .24, st_y[i] - .28, "Station" + str(i + 1), color='k', fontsize=12)

    #ax1.text(0.5, 5.8, "Experiment " + str(exp + 1) + " Heuristic Algorithm", color='k', fontsize=12)
    #ax1.text(0.5, 5.5, "Leader utility : " + str(prob_.leader_utility_history[mi]), color='k', fontsize=12)

    ax1.set_xlim(0, 5.5)
    ax1.set_ylim(0, 6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='upper right', ncol=2, fontsize=16)

    x = np.zeros((5, 4))
    x[:, 0] = np.array([1, 2, 3, 4, 5])
    x[:, 1] = np.round(prob_.leader_decision_history[mi] - 0.5, 1)
    x[:, 2] = prob.target_evs
    x[:, 3] = np.round(np.sum(prob_.followers_decision_history[mi], axis=0), 1)
    print(x)
    df = pd.DataFrame(x, columns=['Station', 'Electricity Price', 'Target EV($V^m$)', 'Expected EV($v^m$)'])
    df['Station'] = df['Station'].apply(int)
    #df['Target EV($V^m$)'] = df['Target EV($V^m$)'].apply(int)
    print(df)
    a = df.values.tolist()
    # a[0][0] = int(a[0][0])
    for i in [0, 2]:
        for j in range(5):
            if i == 0:
                a[j][i] = int(a[j][i])
            if i != 0:
                a[j][i] = np.round(a[j][i], 1)
    ax2 = fig.add_subplot(grid[-1, 0])
    table = ax2.table(cellText=a, colLabels=df.columns, loc='upper center', cellLoc='center',
                      colWidths=[.1, .1, .1, .1], fontsize=100)
    table.auto_set_font_size(False)
    table.scale(2.5, 1.5)
    table.set_fontsize(14)

    # table.scale(1,2)
    ax2.axis('tight')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    """

print(np.average(-np.array(grad)), np.average(-np.array(prox)))
print(np.std(-np.array(grad)), np.std(-np.array(prox)))
plt.figure(figsize=(8,5))
plt.plot(-np.array(grad),color='tab:red', label='Gradient Algorithm')
plt.plot(-np.array(prox),color='tab:green', label='Proximal Algorithm')
plt.legend(fontsize=14)
plt.xlabel("Experiment", fontsize=14)
plt.ylabel("Leader Objective", fontsize=14)
plt.tight_layout()
plt.show()