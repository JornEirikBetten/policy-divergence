import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
import pgx
import haiku as hk
from src.policy_loader.policy_loader import load_rashomon_set, load_best_policies, load_similar_policies
from src.model.model import get_model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import pickle


def create_top_200_policies_plot():
    """Create a plot using only the first 200 policies from each environment"""
    env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]

    # Create filtered dataset from individual performance files
    filtered_data = []

    for env in env_names:
        # Load same-world and different-world policy performances
        same_world_path = os.path.join(os.getcwd(), "evaluation_of_policies", env, "same-world-policy-performances.csv")
        diff_world_path = os.path.join(os.getcwd(), "evaluation_of_policies", env, "different-world-policy-performances.csv")

        # Load same world policies
        same_df = pd.read_csv(same_world_path)
        # Filter for first 200 policies and take their mean rewards
        same_top_200 = same_df[same_df["policy_index"] <= 200][["mean_rewards", "policy_index"]]
        for _, row in same_top_200.iterrows():
            filtered_data.append({
                "environment": env,
                "world": "same",
                "avg_acc_rewards": row["mean_rewards"]
            })

        # Load different world policies
        diff_df = pd.read_csv(diff_world_path)
        # Filter for first 200 policies and take their mean rewards
        diff_top_200 = diff_df[diff_df["policy_index"] <= 200][["mean_rewards", "policy_index"]]
        for _, row in diff_top_200.iterrows():
            filtered_data.append({
                "environment": env,
                "world": "different",
                "avg_acc_rewards": row["mean_rewards"]
            })

    # Create DataFrame from filtered data
    filtered_df = pd.DataFrame(filtered_data)

    # Update env_tags for plot labels
    env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    # Prepare colors
    same_color = "#4C78A8"
    diff_color = "#F58518"

    # Plot each environment in its own subplot
    for i, (env, env_tag) in enumerate(zip(env_names, env_tags)):
        ax = axes[i]

        # Get data for this environment from filtered dataset
        data_same = filtered_df[(filtered_df["environment"] == env) & (filtered_df["world"] == "same")]["avg_acc_rewards"].values
        data_diff = filtered_df[(filtered_df["environment"] == env) & (filtered_df["world"] == "different")]["avg_acc_rewards"].values

        # Create side-by-side boxplots
        bp_same = ax.boxplot(
            [data_same],
            positions=[0],
            widths=0.4,
            patch_artist=True,
            manage_ticks=False,
        )
        bp_diff = ax.boxplot(
            [data_diff],
            positions=[1],
            widths=0.4,
            patch_artist=True,
            manage_ticks=False,
        )

        # Style same world boxes
        for box in bp_same["boxes"]:
            box.set(facecolor=same_color, edgecolor="#333333", alpha=0.7)
        for median in bp_same["medians"]:
            median.set(color="#222222", linewidth=1.5)
        for whisker in bp_same["whiskers"]:
            whisker.set(color="#333333")
        for cap in bp_same["caps"]:
            cap.set(color="#333333")

        # Style different world boxes
        for box in bp_diff["boxes"]:
            box.set(facecolor=diff_color, edgecolor="#333333", alpha=0.7)
        for median in bp_diff["medians"]:
            median.set(color="#222222", linewidth=1.5)
        for whisker in bp_diff["whiskers"]:
            whisker.set(color="#333333")
        for cap in bp_diff["caps"]:
            cap.set(color="#333333")

        # Set subplot labels and formatting
        ax.set_title(env_tag, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["same", "different"], fontsize=10)
        # Only show y-axis label on the leftmost subplot
        if i == 0:
            ax.set_ylabel("average accumulated rewards", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis='y', labelsize=9)

    # Create centered legend above the plots
    legend_handles = [
        Patch(facecolor=same_color, edgecolor="#333333", label="same"),
        Patch(facecolor=diff_color, edgecolor="#333333", label="different"),
    ]
    fig.legend(handles=legend_handles, title="world", bbox_to_anchor=(0.5, 0.98),
               loc='center', ncol=2, fontsize=11, title_fontsize=12)

    plt.tight_layout()
    # Adjust spacing to make room for the legend
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.getcwd() + "/strong-sets/" + "policy_performances_top_200.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# Original code (now as a function that can be called)
env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
# policy_performances = {
#     "environment": [], 
#     "world": [], 
#     "avg_acc_rewards": [], 
# }
# for env_name in env_names:
#     same_world_policy_path = os.getcwd() + "/same-world-policies/" + env_name + "/"
#     different_world_policy_path = os.getcwd() + "/different-world-policies/" + env_name + "/"
#     same_world_policy_performance_path = os.getcwd() + "/evaluation_of_policies/" + env_name + "/same-world-policy-performances.csv"
#     different_world_policy_performance_path = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-policy-performances.csv"
#     different_world_stacked_params, different_world_params_list, different_world_rewards = load_best_policies(different_world_policy_path, different_world_policy_performance_path, 20)
#     same_world_stacked_params, same_world_params_list, same_world_rewards = load_similar_policies(same_world_policy_path, same_world_policy_performance_path, different_world_rewards)
#     print("Different world rewards: ", type(different_world_rewards))
#     print("Same world rewards: ", type(same_world_rewards))
#     for i in range(len(different_world_rewards)): 
#         policy_performances["environment"].append(env_name)
#         policy_performances["world"].append("different")
#         policy_performances["avg_acc_rewards"].append(different_world_rewards[i])
#     for i in range(len(same_world_rewards)): 
#         policy_performances["environment"].append(env_name)
#         policy_performances["world"].append("same")
#         policy_performances["avg_acc_rewards"].append(same_world_rewards[i])
    
#     stacked_params_save_path = os.getcwd() + "/strong-sets/" + env_name + "/"
#     if not os.path.exists(stacked_params_save_path):
#         os.makedirs(stacked_params_save_path)
#     with open(stacked_params_save_path + "same-world-stacked-params.pkl", "wb") as f:
#         pickle.dump(same_world_stacked_params, f)
#     with open(stacked_params_save_path + "different-world-stacked-params.pkl", "wb") as f:
#         pickle.dump(different_world_stacked_params, f)
        
policy_performances_save_path = os.getcwd() + "/strong-sets/" + "policy_performances.csv"
# df = pd.DataFrame(policy_performances)
# df.to_csv(policy_performances_save_path, index=False)
df = pd.read_csv(policy_performances_save_path)
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
plt.style.use("seaborn-v0_8")
fig, axes = plt.subplots(1, 5, figsize=(15, 5))

# Prepare colors
same_color = "#4C78A8"
diff_color = "#F58518"

# Plot each environment in its own subplot
for i, (env, env_tag) in enumerate(zip(env_names, env_tags)):
    ax = axes[i]

    # Get data for this environment
    data_same = df[(df["environment"] == env) & (df["world"] == "same")]["avg_acc_rewards"].values
    data_diff = df[(df["environment"] == env) & (df["world"] == "different")]["avg_acc_rewards"].values

    # Create side-by-side boxplots
    bp_same = ax.boxplot(
        [data_same],
        positions=[0],
        widths=0.4,
        patch_artist=True,
        manage_ticks=False,
    )
    bp_diff = ax.boxplot(
        [data_diff],
        positions=[1],
        widths=0.4,
        patch_artist=True,
        manage_ticks=False,
    )

    # Style same world boxes
    for box in bp_same["boxes"]:
        box.set(facecolor=same_color, edgecolor="#333333", alpha=0.7)
    for median in bp_same["medians"]:
        median.set(color="#222222", linewidth=1.5)
    for whisker in bp_same["whiskers"]:
        whisker.set(color="#333333")
    for cap in bp_same["caps"]:
        cap.set(color="#333333")

    # Style different world boxes
    for box in bp_diff["boxes"]:
        box.set(facecolor=diff_color, edgecolor="#333333", alpha=0.7)
    for median in bp_diff["medians"]:
        median.set(color="#222222", linewidth=1.5)
    for whisker in bp_diff["whiskers"]:
        whisker.set(color="#333333")
    for cap in bp_diff["caps"]:
        cap.set(color="#333333")

    # Set subplot labels and formatting
    ax.set_title(env_tag, fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["same", "different"], fontsize=10)
    # Only show y-axis label on the leftmost subplot
    if i == 0:
        ax.set_ylabel("average accumulated rewards", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis='y', labelsize=9)

# Create centered legend above the plots
legend_handles = [
    Patch(facecolor=same_color, edgecolor="#333333", label="same"),
    Patch(facecolor=diff_color, edgecolor="#333333", label="different"),
]
fig.legend(handles=legend_handles, title="world", bbox_to_anchor=(0.5, 0.98),
           loc='center', ncol=2, fontsize=11, title_fontsize=12)

plt.tight_layout()
# Adjust spacing to make room for the legend
plt.subplots_adjust(top=0.85)
plt.savefig(os.getcwd() + "/strong-sets/" + "policy_performances.pdf", format="pdf", bbox_inches="tight")


# Call the new function to create the top 200 policies plot
if __name__ == "__main__":
    create_top_200_policies_plot()