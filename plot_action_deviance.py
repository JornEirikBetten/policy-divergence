import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.ticker import MaxNLocator
path_to_action_deviance_data = os.getcwd() + "/interpretation_data/"
action_deviance_data = pd.read_csv(path_to_action_deviance_data + "action_deviance_data.csv")

envs = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
training_setups = ["DWDI", "SWDI", "DWSI"]
training_labels = ["B", "AD", "ED"]
num_eval_envs = 1024

colors = ['#2E86AB', '#A23B72', '#F18F01']
fig, axs = plt.subplots(2, 5, figsize=(15, 5))
second_fig, second_axs = plt.subplots(2, 5, figsize=(15, 5))
for i, (env, env_tag) in enumerate(zip(envs, env_tags)):
    ax = axs[0, i]
    second_ax = second_axs[0, i]
    max_action_deviance = 0
    violin_data = []
    for j, training_setup in enumerate(training_setups):
        x_pos = 0.5*(j+1)
        data = action_deviance_data[(action_deviance_data["environment"] == env) & (action_deviance_data["training_setup"] == training_setup)]
        fraction_action_deviance = data["fraction_action_deviance"].values
        max_action_deviance = max(max_action_deviance, fraction_action_deviance.max())
        ax.boxplot(fraction_action_deviance, positions=[x_pos], widths=0.4, patch_artist=True, boxprops=dict(facecolor=colors[j], alpha=0.7, edgecolor='black', linewidth=0.5), 
                   medianprops=dict(color='black', linewidth=1.0), whiskerprops=dict(color='black', linewidth=0.5))
        violin_data.append(fraction_action_deviance)
    second_ax.set_xlim(-1, 3)
    parts = second_ax.violinplot(violin_data, positions=range(3), widths=0.5, showmeans=True)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.0)
    ax.set_title(f'{env_tag}', fontsize=16, fontweight='semibold', pad=10, fontfamily='sans-serif', color='#34495E')
    second_ax.set_title(f'{env_tag}', fontsize=16, fontweight='semibold', pad=10, fontfamily='sans-serif', color='#34495E')
    if i == 0:
        ax.set_ylabel("fraction of action deviance\n (in all states)", multialignment='center', fontsize=10)
        second_ax.set_ylabel("fraction of action deviance\n (in all states)", multialignment='center', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    second_ax.spines['top'].set_visible(False)
    second_ax.spines['right'].set_visible(False)
    second_ax.spines['left'].set_linewidth(1.0)
    second_ax.spines['bottom'].set_linewidth(1.0)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels(training_labels, fontsize=10)
    ax.set_ylim(0, max_action_deviance+0.02)
    ax.set_yticks(np.arange(0, max_action_deviance+0.02, (max_action_deviance+0.02)/5))
    ax.set_yticklabels([f"{y*100:.1f}%" for y in ax.get_yticks()])
    second_ax.set_ylim(0, max_action_deviance+0.02)
    second_ax.set_yticks(np.arange(0, max_action_deviance+0.02, (max_action_deviance+0.02)/5))
    second_ax.set_yticklabels([f"{y*100:.1f}%" for y in second_ax.get_yticks()])
    second_ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    second_ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    second_ax.set_xticks([0, 1, 2])
    second_ax.set_xticklabels(training_labels, fontsize=10)
    #second_ax.set_xlabel("training context", fontsize=10)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)

for i, (env, env_tag) in enumerate(zip(envs, env_tags)):
    ax = axs[1, i]
    second_ax = second_axs[1, i]
    max_action_deviance = 0
    violin_data = []
    for j, training_setup in enumerate(training_setups):
        x_pos = 0.5*(j+1)
        data = action_deviance_data[(action_deviance_data["environment"] == env) & (action_deviance_data["training_setup"] == training_setup)]
        fraction_action_deviance_in_most_important_states = data["fraction_action_deviance_in_most_important_states"].values
        max_action_deviance = max(max_action_deviance, fraction_action_deviance_in_most_important_states.max())
        ax.boxplot(fraction_action_deviance_in_most_important_states, positions=[x_pos], widths=0.4, patch_artist=True, boxprops=dict(facecolor=colors[j], alpha=0.7, edgecolor='black', linewidth=0.5), 
                   medianprops=dict(color='black', linewidth=1.0), whiskerprops=dict(color='black', linewidth=0.5))
        violin_data.append(fraction_action_deviance_in_most_important_states)
    second_ax.set_xlim(-1, 3)
    parts = second_ax.violinplot(violin_data, positions=range(3), widths=0.5, showmeans=True)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)
    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1.0)
    if i == 0:
        ax.set_ylabel("fraction of action deviance\n (in most critical states)", multialignment='center', fontsize=10)
        second_ax.set_ylabel("fraction of action deviance\n (in most critical states)", multialignment='center', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    second_ax.spines['top'].set_visible(False)
    second_ax.spines['right'].set_visible(False)
    second_ax.spines['left'].set_linewidth(1.0)
    second_ax.spines['bottom'].set_linewidth(1.0)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels(training_labels, fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    ax.set_ylim(0, max_action_deviance+0.005)
    ax.set_yticks(np.arange(0, max_action_deviance+0.005, (max_action_deviance+0.005)/5))
    ax.set_yticklabels([f"{y*100:.1f}%" for y in ax.get_yticks()])
    #ax.set_yticks()
    second_ax.set_ylim(0, max_action_deviance+0.005)
    second_ax.set_yticks(np.arange(0, max_action_deviance+0.005, (max_action_deviance+0.005)/5))
    second_ax.set_yticklabels([f"{y*100:.1f}%" for y in second_ax.get_yticks()])
    second_ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    second_ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    second_ax.set_xticks([0, 1, 2])
    second_ax.set_xticklabels(training_labels, fontsize=10)
    second_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    second_ax.set_ylim(0, max_action_deviance+0.005)
    second_ax.set_yticks(np.arange(0, max_action_deviance+0.005, (max_action_deviance+0.005)/5))
    second_ax.set_yticklabels([f"{y*100:.1f}%" for y in second_ax.get_yticks()])
    second_ax.set_xlabel("training context", fontsize=10)
    
handles, labels = axs[1, 0].get_legend_handles_labels()
second_handles, second_labels = second_axs[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
second_fig.legend(second_handles, second_labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)    
fig.savefig(os.getcwd() + "/interpretation_data/action_deviance_plot.pdf", format="pdf", bbox_inches="tight")
#fig.close()
second_fig.savefig(os.getcwd() + "/interpretation_data/action_deviance_plot_violin.pdf", format="pdf", bbox_inches="tight")
#second_fig.close()