import pandas as pd 
import os 
import numpy as np 
from src.policy_loader.policy_loader import load_best_policies, load_similar_policies
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
path_to_ensemble_performance_data = os.getcwd() + "/evaluation_of_policies/mv_ensemble_results.csv"
ensemble_performance_data = pd.read_csv(path_to_ensemble_performance_data)

env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
training_setups = ["different-world", "same-world", "different-world-same-init"]
labels = ["B", "AD", "ED"]
training_labels = ["DWDI", "SWDI", "DWSI"]
num_eval_envs = 1024

def load_policies(env_name):
    path_to_policies = os.getcwd() + "/different-world-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-policy-performances.csv"
    different_world_params_stacked, different_world_params_list, different_world_rewards = load_best_policies(path_to_policies, path_to_evaluation_data, 20)
    print("different_world_rewards: ", different_world_rewards)
    path_to_policies = os.getcwd() + "/same-world-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/same-world-policy-performances.csv"
    same_world_params_stacked, same_world_params_list, same_world_rewards = load_similar_policies(path_to_policies, path_to_evaluation_data, different_world_rewards, "swdi")
    print("same_world_rewards: ", same_world_rewards)
    path_to_policies = os.getcwd() + "/different-world-same-init-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-same-init-policy-performances.csv"
    different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards = load_similar_policies(path_to_policies, path_to_evaluation_data, different_world_rewards, "dwsi")
    print("different-world-same-init-rewards: ", different_world_same_init_rewards)
    return different_world_rewards, same_world_rewards, different_world_same_init_rewards


relative_performances = {
    "environment": [],
    "context": [],
    "relative_performance_increase_mean": [], 
    "relative_performance_increase_std": []
}
colors = ['#2E86AB', '#A23B72', '#F18F01']
fig, axs = plt.subplots(1, 5, figsize=(15, 4))
for i, (env_tag, env_name) in enumerate(zip(env_tags, env_names)):
    different_world_rewards, same_world_rewards, different_world_same_init_rewards = load_policies(env_name)
    max_rewards = [np.max(different_world_rewards), np.max(same_world_rewards), np.max(different_world_same_init_rewards)]
    #print(max_rewards)
    violin_data = [different_world_rewards, same_world_rewards, different_world_same_init_rewards]
    ax = axs[i]
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
    ax.set_xticklabels(labels, fontsize=10)#, fontweight='semibold')
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
    for i, training_setup in enumerate(training_setups):
        # individual_performance_path = os.getcwd() + f"/evaluation_of_policies/{env_name}/{training_setup}-policy-performances.csv"
        # individual_performance_data = pd.read_csv(individual_performance_path)
        # policy_path = os.getcwd() + f"/different-world-policies/{env_name}/{training_setup}-policy-performances.csv"
        # individual_performance = individual_performance_data["mean_rewards"].values
        # best_individual_performance = individual_performance.max()
        ensemble_performance = ensemble_performance_data[(ensemble_performance_data["env_name"] == env_name) & (ensemble_performance_data["policy_type"] == training_setup)]["mean_rewards"].values
        ensemble_stds = ensemble_performance_data[(ensemble_performance_data["env_name"] == env_name) & (ensemble_performance_data["policy_type"] == training_setup)]["std_rewards"].values
        print(ensemble_performance)
        best_individual_performance = max_rewards[i]
        performance_increase_mean = ensemble_performance / best_individual_performance
        performance_increase_std = ensemble_stds / best_individual_performance
        print(f"{env_name}: {labels[i]}: {performance_increase_mean}")
        relative_performances["environment"].append(env_name)
        relative_performances["context"].append(labels[i])
        relative_performances["relative_performance_increase_mean"].append(performance_increase_mean[0])
        relative_performances["relative_performance_increase_std"].append(performance_increase_std[0])
fig.savefig(os.getcwd() + f"/evaluation_of_policies/ensemble_comparison_violin_plots.pdf", format="pdf", bbox_inches="tight")
plt.close()
df = pd.DataFrame(relative_performances)
df.to_csv(os.getcwd() + "/evaluation_of_policies/relative_performance_increases.csv", index=False)