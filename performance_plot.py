import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import os
path_to_ensemble_results = os.getcwd() + "/evaluation_of_policies/mv_ensemble_results.csv"
ensemble_results = pd.read_csv(path_to_ensemble_results)


envs = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
training_setups = ["different-world", "same-world", "different-world-same-init"]
training_labels = ["B", "AD", "ED"]
num_eval_envs = 1024

colors = ['#2E86AB', '#A23B72', '#F18F01']
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
for i, (env, env_tag) in enumerate(zip(envs, env_tags)):
    ax = axs[i]
    for j, training_setup in enumerate(training_setups):
        data = ensemble_results[(ensemble_results["env_name"] == env) & (ensemble_results["policy_type"] == training_setup)]
        mean_rewards = data["mean_rewards"].values
        x_pos = j
        std_rewards = data["std_rewards"].values
        ax.errorbar(x_pos, mean_rewards, yerr=std_rewards/np.sqrt(num_eval_envs), 
                    label=training_labels[j], color=colors[j], marker='o', linestyle='-', markersize=12, linewidth=2, 
                    markeredgecolor='black', markeredgewidth=1.0)
    ax.set_xlim(-1, 3)
    ax.set_title(f'{env_tag}',
                        fontsize=16, fontweight='semibold', pad=10,
                        fontfamily='sans-serif', color='#34495E')
    if i == 0:
        ax.set_ylabel("average accumulated rewards\n (majority-vote ensembles)", multialignment='center', fontsize=10)

    # if i == 4:
    #     ax.legend(fontsize=12, loc="upper left")
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    #ax.spines['bottom'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    ax.set_xticks(range(3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    ax.set_xticklabels(training_labels, fontsize=10)
    
    #ax.set_yticks()
    
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
plt.savefig(os.getcwd() + "/evaluation_of_policies/ensemble_performances.pdf", format="pdf", bbox_inches="tight")
plt.close()
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
for i, (env, env_tag) in enumerate(zip(envs, env_tags)):
    ax = axs[i]
    path_to_dwdi_results = os.getcwd() + f"/evaluation_of_policies/{env}/different-world-policy-performances.csv"
    path_to_swdi_results = os.getcwd() + f"/evaluation_of_policies/{env}/same-world-policy-performances.csv"
    path_to_dwsi_results = os.getcwd() + f"/evaluation_of_policies/{env}/different-world-same-init-policy-performances.csv"
    dwdi_results = pd.read_csv(path_to_dwdi_results)
    swdi_results = pd.read_csv(path_to_swdi_results)
    dwsi_results = pd.read_csv(path_to_dwsi_results) 
    violin_data = [dwdi_results["mean_rewards"].values[:200], swdi_results["mean_rewards"].values[:200], dwsi_results["mean_rewards"].values[:200]]
    ax.set_xlim(-1, 3)    
        # Create violin plot
    parts = ax.violinplot(violin_data, positions=range(3),
                                    showmeans=True, widths=0.5)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.0)

    for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.0)
    # Styling for violin plot
    ax.set_xlabel('training context', fontsize=12)
    ax.set_title(f'{env_tag}',
                    fontsize=16, fontweight='semibold', pad=10,
                    fontfamily='sans-serif', color='#34495E')
    ax.set_xticks(range(3))
    ax.set_xticklabels(training_labels, fontsize=10)#, fontweight='semibold')
    #ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5)
    #ax.set_ylim(0, 1)
    if i == 0:
        ax.set_ylabel("average accumulated rewards", fontsize=12)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
plt.savefig(os.getcwd() + f"/evaluation_of_policies/performance_comparison_violin_plots.pdf", format="pdf", bbox_inches="tight")



path_to_relative_performance_increases = os.getcwd() + "/evaluation_of_policies/relative_performance_increases.csv"
relative_performance_increases = pd.read_csv(path_to_relative_performance_increases)
print(relative_performance_increases.head())
for env in envs:
    env_data = relative_performance_increases[relative_performance_increases["environment"] == env]
    #print(env_data.head())
    for context in ["B", "AD", "ED"]:
        context_data = env_data[env_data["context"] == context]
        mean_relative_performance_increase = context_data["relative_performance_increase_mean"].values[0]
        std_relative_performance_increase = context_data["relative_performance_increase_std"].values[0]
        print(f"Environment: {env}, Context: {context}, Mean relative performance increase: {mean_relative_performance_increase}, Std relative performance increase: {std_relative_performance_increase}")
        std_error_of_mean = std_relative_performance_increase / np.sqrt(num_eval_envs)
        print(f"Std error of mean: {std_error_of_mean}")



fig, axs = plt.subplots(1, 5, figsize=(15, 4))
for i, (env, env_tag) in enumerate(zip(envs, env_tags)):
    ax = axs[i]
    for j, context in enumerate(["B", "AD", "ED"]):
        data = relative_performance_increases[(relative_performance_increases["environment"] == env) & (relative_performance_increases["context"] == context)]
        mean_relative_performance_increase = data["relative_performance_increase_mean"].values[0]
        print(f"Environment: {env}, Context: {context}, Mean relative performance increase: {mean_relative_performance_increase}")
        std_relative_performance_increase = data["relative_performance_increase_std"].values[0]
        std_error_of_mean = std_relative_performance_increase / np.sqrt(num_eval_envs)
        # print(f"Environment: {env}, Context: {context}, Mean relative performance increase: {mean_relative_performance_increase}, Std relative performance increase: {std_relative_performance_increase}")
        # print(f"Std error of mean: {std_error_of_mean}")
        ax.errorbar(j, mean_relative_performance_increase, yerr=std_error_of_mean, 
                    label=context, color=colors[j], marker='o', linestyle='-', markersize=12, linewidth=2, 
                    markeredgecolor='black', markeredgewidth=1.0)
    ax.set_xlim(-1, 3)
    ax.set_title(f'{env_tag}',
                        fontsize=16, fontweight='semibold', pad=10,
                        fontfamily='sans-serif', color='#34495E')
    if i == 0:
        ax.set_ylabel("relative performance increase\n (majority-vote ensembles)", multialignment='center', fontsize=10)

    # if i == 4:
    #     ax.legend(fontsize=12, loc="upper left")
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    #ax.spines['bottom'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    ax.tick_params(axis='y', which='major', labelsize=8, width=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(training_labels, fontsize=10) 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
    ax.set_yticklabels([f"{y*100:.1f}%" for y in ax.get_yticks()])
    ax.set_xlabel("training context", fontsize=10)
    #ax.set_yticks()
    
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
plt.savefig(os.getcwd() + "/evaluation_of_policies/relative_ensemble_performances.pdf", format="pdf", bbox_inches="tight")
plt.close()
# from statistical_tests import print_comparison_report
# envs = ["minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
# for env in envs:
#     dwdi_results = pd.read_csv(os.getcwd() + f"/evaluation_of_policies/{env}/different-world-policy-performances.csv")
#     swdi_results = pd.read_csv(os.getcwd() + f"/evaluation_of_policies/{env}/same-world-policy-performances.csv")
#     dwsi_results = pd.read_csv(os.getcwd() + f"/evaluation_of_policies/{env}/different-world-same-init-policy-performances.csv")
#     print(f"Environment: {env}")
#     print_comparison_report(dwdi_results["mean_rewards"].values, swdi_results["mean_rewards"].values, label1="DWDI", label2="SWDI")
#     print_comparison_report(dwdi_results["mean_rewards"].values, dwsi_results["mean_rewards"].values, label1="DWDI", label2="DWSI")
#     print_comparison_report(swdi_results["mean_rewards"].values, dwsi_results["mean_rewards"].values, label1="SWDI", label2="DWSI")