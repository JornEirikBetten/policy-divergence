import jax
#jax.config.update('jax_platform_name', 'cpu') # Using CPU because of the memory load
import jax.numpy as jnp 
import haiku as hk 
import chex 
import pgx 
from typing import NamedTuple, Callable, Tuple
from dataclasses import fields 
import os 
import sys 
import pickle 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as colors
from src.policy_loader.policy_loader import load_best_policies, load_similar_policies
from src.model.model import get_model
import seaborn as sns

class Top5Features(NamedTuple): 
    dwdi_top_5_features: chex.Array
    swdi_top_5_features: chex.Array
    dwsi_top_5_features: chex.Array
    


def build_saliency_map_fn(env_name):
    env_fn = pgx.make(env_name)
    forward_pass = get_model(env_fn.num_actions)
    forward_pass = hk.without_apply_rng(hk.transform(forward_pass)) 
    def saliency_map_fn(stacked_params, observation): 
        """A function that returns the saliency map for a single observation.
        stacked_params: (num_policies, architecture)
        observation: (height, width, num_channels)
        Returns: (num_policies, height, width, num_channels)
        """
        # Reshape to match with parameters
        observation = observation.astype(jnp.float32).reshape((1,) + observation.shape)
        logits, value = forward_pass.apply(stacked_params, observation)
        summed_forward_function = lambda obs: jnp.sum(forward_pass.apply(stacked_params, obs)[0], axis=-1)[0]
        saliency_map = jax.grad(summed_forward_function)(observation)
        # Mean-max scaling of saliency map
        #saliency_map = saliency_map.astype(jnp.float32)
        abs_saliency_map = jnp.abs(saliency_map)
        scaled_saliency_map = (abs_saliency_map - jnp.min(abs_saliency_map)) / (jnp.max(abs_saliency_map) - jnp.min(abs_saliency_map))
        return stacked_params, scaled_saliency_map
    
    return saliency_map_fn

def build_plot_image_of_observation_fn(cmap, norm): 
    #plt.style.use("seaborn-v0_8-darkgrid")
    def plot_image_of_observation(observation, path_to_save): 
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        numerical_obs = jnp.amax(observation* jnp.reshape(jnp.arange(observation.shape[-1]) + 1, (1, 1, -1)), 2) + 0.5 
        ax.imshow(numerical_obs, 
            aspect="equal", 
            cmap=cmap,
            norm=norm)
        plt.savefig(path_to_save, format="pdf", bbox_inches="tight")
        plt.close()
    return plot_image_of_observation

def plot_saliency_map(saliency_map, path_to_save): 
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    saliency = jnp.sum(saliency_map, axis=-1)
    ax.imshow(saliency, 
        aspect="equal", 
        cmap=cm.binary)
    plt.savefig(path_to_save, format="pdf", bbox_inches="tight")
    plt.close()
env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]

similarities = {
    "environment": [], 
    "mean_similarity": [],
    "sem_similarity": [], 
    "n_observations": [], 
    "training_setup": [],
}
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
for env_name, env_tag in zip(env_names, env_tags):
    #env_name = sys.argv[1]
    trajectory_path = os.getcwd() + "/trajectories/" + env_name + "/same-world-trajectories.pkl"

    with open(trajectory_path, "rb") as f:
        trajectories = pickle.load(f)
    env_fn = pgx.make(env_name)
    path_to_policies = os.getcwd() + "/different-world-policies/" + env_name + "/"
    path_to_same_world_policies = os.getcwd() + "/same-world-policies/" + env_name + "/"
    path_to_different_world_same_init_policies = os.getcwd() + "/different-world-same-init-policies/" + env_name + "/"
    path_to_same_world_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/same-world-policy-performances.csv"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-policy-performances.csv"
    path_to_different_world_same_init_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-same-init-policy-performances.csv"
    num_policies = 20
    different_world_params_stacked, different_world_params_list, different_world_rewards = load_best_policies(path_to_policies, path_to_evaluation_data, num_policies)
    same_world_params_stacked, same_world_params_list, same_world_rewards = load_similar_policies(path_to_same_world_policies, path_to_same_world_evaluation_data, different_world_rewards, "swdi")
    different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards = load_similar_policies(path_to_different_world_same_init_policies, path_to_different_world_same_init_evaluation_data, different_world_rewards, "dwsi")
    save_path = os.getcwd() + "/feature_analysis/" + env_name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saliency_map_fn = build_saliency_map_fn(env_name)
    multiple_params_saliency_map_fn = jax.vmap(saliency_map_fn, in_axes=(0, None))
    




    def extract_five_most_important_features(saliencies):
        """
        Extract the five most important features from the saliency map.
        saliencies: (num_policies, num_features)
        Returns: (num_policies, 5)
        """
        #print("Saliencies: ", saliencies.shape)
        extract_top_5 = lambda x: jnp.argsort(x)[-5:]
        most_important_features = jax.vmap(extract_top_5, in_axes=(0))(saliencies)
        #print("Most important features: ", most_important_features.shape)
        return most_important_features



    def compute_feature_similarity_kernel(most_important_features):
        """
        Compute similarity between policies based on their top features.
        most_important_features: (num_policies, 5) - indices of top 5 features for each policy
        Returns: (num_policies, num_policies) - similarity matrix where entry (i,j) is the number
                of common features between policy i and policy j, normalized by 5
        """
        # print("Most important features: ", most_important_features.shape)
        #triu_indices = jnp.triu_indices(most_important_features.shape[0], k=1)
        #Vectorized computation of pairwise feature overlap
        # def count_common_features(carry, pair):
        #     # features_i: (5,) - top features for policy i
        #     # features_j: (5,) - top features for policy j
        #     # Returns: scalar - number of common features (0-5)
        #     overlap = jax.vmap(jnp.isin, in_axes=(0, None))(pair[0], pair[1])
        #     #print("Overlap: ", overlap.shape)
        #     return carry, jnp.sum(overlap)
        def count_common_features(f1, f2): 
            overlap = jnp.sum(jax.vmap(jnp.isin, in_axes=(0, None))(f1, f2))
            return overlap
            
        common_features = jax.vmap(jax.vmap(count_common_features, in_axes=(0, None)), in_axes=(None, 0))(most_important_features, most_important_features)
        return common_features / 5.0



    def extract_top_features(carry, observations):
        parameters = carry
        # Get saliency
        saliency_map_fn = build_saliency_map_fn(env_name)
        multiple_params_saliency_map_fn = jax.vmap(saliency_map_fn, in_axes=(0, None))
        params, saliencies = jax.lax.scan(
            multiple_params_saliency_map_fn,
            parameters,
            xs=observations
        )
        saliencies = jnp.reshape(saliencies, (saliencies.shape[0], saliencies.shape[1], -1))
        most_important_features = jax.vmap(extract_five_most_important_features, in_axes=(1))(saliencies)
        # Compute feature similarity kernel (pairwise similarity between policies)
        similarity_matrix = jax.vmap(compute_feature_similarity_kernel, in_axes=(1))(most_important_features)
        # For each policy, compute average similarity with all other policies
        # This gives us a measure of how similar each policy's feature importance is to others
        # mean_similarity = jnp.mean(similarity_matrix, axis=-1)
        # print("Mean similarity per policy: ", mean_similarity.shape)
        return parameters, similarity_matrix



    #most_important_features = jax.vmap(extract_top_features, in_axes=(None, 0))(different_world_params_stacked, trajectories.state.observation)
    # params, similarity_matrices = jax.lax.scan(
    #     f=extract_top_features,
    #     init=different_world_params_stacked,
    #     xs=trajectories.state.observation[:10, :10, :, :, :]
    # )
    different_world_params_stacked, swdi_similarity_matrices = jax.lax.scan(
        extract_top_features, 
        different_world_params_stacked, 
        xs=trajectories.state.observation
    )
    n_observations = swdi_similarity_matrices.shape[0]*swdi_similarity_matrices.shape[1]
    mean_swdi_similarity_matrices = jnp.mean(swdi_similarity_matrices, axis=(0, 1))
    std_swdi_similarity_matrices = jnp.std(swdi_similarity_matrices, axis=(0, 1))
    sem_swdi_similarity_matrices = std_swdi_similarity_matrices / jnp.sqrt(swdi_similarity_matrices.shape[0]*swdi_similarity_matrices.shape[1])
    similarities["environment"].append(env_tag)
    similarities["training_setup"].append("different-world")
    similarities["mean_similarity"].append(mean_swdi_similarity_matrices)
    similarities["sem_similarity"].append(sem_swdi_similarity_matrices)
    similarities["n_observations"].append(n_observations)
    same_world_params_stacked, dwsi_similarity_matrices = jax.lax.scan(
        extract_top_features, 
        same_world_params_stacked, 
        xs=trajectories.state.observation
    )
    mean_dwsi_similarity_matrices = jnp.mean(dwsi_similarity_matrices, axis=(0, 1))
    std_dwsi_similarity_matrices = jnp.std(dwsi_similarity_matrices, axis=(0, 1))
    sem_dwsi_similarity_matrices = std_dwsi_similarity_matrices / jnp.sqrt(dwsi_similarity_matrices.shape[0]*dwsi_similarity_matrices.shape[1])
    similarities["environment"].append(env_tag)
    similarities["training_setup"].append("same-world")
    similarities["mean_similarity"].append(mean_dwsi_similarity_matrices)
    similarities["sem_similarity"].append(sem_dwsi_similarity_matrices)
    similarities["n_observations"].append(n_observations)
    different_world_same_init_params_stacked, dwsi_similarity_matrices = jax.lax.scan(
        extract_top_features, 
        different_world_same_init_params_stacked, 
        xs=trajectories.state.observation
    )
    mean_dwsi_similarity_matrices = jnp.mean(dwsi_similarity_matrices, axis=(0, 1))
    std_dwsi_similarity_matrices = jnp.std(dwsi_similarity_matrices, axis=(0, 1))
    sem_dwsi_similarity_matrices = std_dwsi_similarity_matrices / jnp.sqrt(dwsi_similarity_matrices.shape[0]*dwsi_similarity_matrices.shape[1])
    similarities["environment"].append(env_tag)
    similarities["training_setup"].append("different-world-same-init")
    similarities["mean_similarity"].append(mean_dwsi_similarity_matrices)
    similarities["sem_similarity"].append(sem_dwsi_similarity_matrices)
    similarities["n_observations"].append(n_observations)
    
    

    # print("Similarity matrices: ", similarity_matrices.shape)

    # # Compute mean similarity matrix across all observations
    # mean_similarity_matrix = jnp.mean(similarity_matrices, axis=(0, 1))  # Shape: (num_policies, num_policies)

    # # Compute standard error of the mean (SEM) for each policy pair
    # # SEM = std / sqrt(n) where n is number of observations
    # std_similarity_matrix = jnp.std(similarity_matrices, axis=(0, 1))  # Standard deviation across observations
    
    # sem_similarity_matrix = std_similarity_matrix / jnp.sqrt(n_observations)  # Standard error of the mean

    # print("Mean similarity matrix shape:", mean_similarity_matrix.shape)
    # print("SEM matrix shape:", sem_similarity_matrix.shape)

    # similarities["environment"].append(env_tag)
    # similarities["mean_similarity"].append(mean_similarity_matrix)
    # similarities["sem_similarity"].append(sem_similarity_matrix)
    # similarities["n_observations"].append(n_observations)

save_path = os.getcwd() + "/feature_analysis/similarities.pkl"
with open(save_path, "wb") as f:
    pickle.dump(similarities, f)
    
save_path = os.getcwd() + "/feature_analysis/similarities.pkl"
similarities = pickle.load(open(save_path, "rb"))
for env in env_tags: 
    for training_setup in ["different-world", "same-world", "different-world-same-init"]:
        mean_similarity = similarities["mean_similarity"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
        sem_similarity = similarities["sem_similarity"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
        n_observations = similarities["n_observations"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
        print(f"Environment: {env}, Training setup: {training_setup}, Mean similarity: {mean_similarity}, SEM similarity: {sem_similarity}, N observations: {n_observations}")
        
        
def make_figure(similarities, save_path): 
    """Make a figure of the similarity matrix."""
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, env in enumerate(env_tags):
        for j, training_setup in enumerate(["different-world", "same-world", "different-world-same-init"]):
            mean_similarity = similarities["mean_similarity"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
            sem_similarity = similarities["sem_similarity"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
            n_observations = similarities["n_observations"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
            axes[j, i].imshow(mean_similarity, cmap=cm.binary, vmin=0, vmax=1, alpha=0.9)
            axes[j, i].set_title(f"{env}, {training_setup}")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    return fig
path_to_save = os.getcwd() + "/feature_analysis/similarity_matrices.pdf"
make_figure(similarities, path_to_save)
# # Visualize the mean similarity matrix as a heatmap
# def plot_similarity_heatmap(mean_matrix, sem_matrix, save_path, title="Policy Similarity Matrix"):
#     """Plot heatmap of mean similarity matrix with SEM error bars"""
#     # Set up a more aesthetic style
#     plt.style.use('default')  # Reset to default first

#     # Create figure with better proportions
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))

#     # Use a more sophisticated colormap
#     main_cmap = 'RdYlBu_r'  # Red-Yellow-Blue reversed (blue=low, red=high)

#     # Main heatmap of mean similarities with cell values
#     im1 = axes[0, 0].imshow(mean_matrix, cmap=main_cmap, vmin=0, vmax=1, alpha=0.9)

#     # Add cell values as text with very small font
#     for i in range(mean_matrix.shape[0]):
#         for j in range(mean_matrix.shape[1]):
#             # Format the number to 2 decimal places
#             value = f'{mean_matrix[i, j]:.2f}'
#             # Add text with very small font, centered in cell
#             axes[0, 0].text(j, i, value,
#                         ha='center', va='center',
#                         color='white' if mean_matrix[i, j] > 0.5 else 'black',
#                         fontweight='bold',
#                         fontsize=6)  # Very small font

#     axes[0, 0].set_title(f'{title}\n(Mean across {n_observations} observations)',
#                         fontsize=14, fontweight='bold', pad=20)
#     axes[0, 0].set_xlabel('Policy Index', fontsize=12)
#     axes[0, 0].set_ylabel('Policy Index', fontsize=12)
#     axes[0, 0].tick_params(axis='both', which='major', labelsize=10)

#     # Colorbar for main heatmap
#     cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, shrink=0.8)
#     cbar1.ax.tick_params(labelsize=10)
#     cbar1.set_label('Similarity', rotation=270, labelpad=15, fontsize=12)

#     # Heatmap of SEM values (no cell values needed for SEM)
#     sem_cmap = 'YlOrRd'  # Yellow-Orange-Red for SEM (higher = more uncertainty)
#     im2 = axes[0, 1].imshow(sem_matrix, cmap=sem_cmap, vmin=0, alpha=0.9)

#     axes[0, 1].set_title('Standard Error of the Mean (SEM)',
#                         fontsize=14, fontweight='bold', pad=20)
#     axes[0, 1].set_xlabel('Policy Index', fontsize=12)
#     axes[0, 1].set_ylabel('Policy Index', fontsize=12)
#     axes[0, 1].tick_params(axis='both', which='major', labelsize=10)

#     # Colorbar for SEM heatmap
#     cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04, shrink=0.8)
#     cbar2.ax.tick_params(labelsize=10)
#     cbar2.set_label('SEM', rotation=270, labelpad=15, fontsize=12)

#     # Diagonal plot: mean similarity vs policy index - make it more aesthetic
#     policy_indices = jnp.arange(mean_matrix.shape[0])
#     mean_diagonal = jnp.diag(mean_matrix)
#     sem_diagonal = jnp.diag(sem_matrix)

#     # Create a more visually appealing errorbar plot
#     axes[1, 0].errorbar(policy_indices, mean_diagonal, yerr=sem_diagonal,
#                     fmt='o-', capsize=5, alpha=0.8, linewidth=2,
#                     markersize=6, color='#2E86AB', ecolor='#A23B72')

#     # Styling for diagonal plot
#     axes[1, 0].set_title('Self-Similarity (Diagonal Values)',
#                         fontsize=14, fontweight='bold', pad=20)
#     axes[1, 0].set_xlabel('Policy Index', fontsize=12)
#     axes[1, 0].set_ylabel('Similarity', fontsize=12)
#     axes[1, 0].grid(True, alpha=0.2, linestyle='--', color='gray')
#     axes[1, 0].axhline(y=1.0, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2,
#                     label='Perfect similarity (1.0)')
#     axes[1, 0].axhspan(0.8, 1.0, alpha=0.1, color='green', label='High similarity (>0.8)')
#     axes[1, 0].tick_params(axis='both', which='major', labelsize=10)

#     # Enhanced legend
#     axes[1, 0].legend(fontsize=10, loc='lower right')

#     # Histogram of all similarity values - make it more aesthetic
#     all_similarities = mean_matrix.flatten()

#     # Create histogram with better styling
#     n, bins, patches = axes[1, 1].hist(all_similarities, bins=20, alpha=0.7,
#                                     edgecolor='black', linewidth=1.2)

#     # Color the histogram bars based on value ranges
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     colors_list = ['#E74C3C', '#F39C12', '#F7DC6F', '#82E0AA', '#28B463']  # Red to Green
#     for i, patch in enumerate(patches):
#         if bin_centers[i] < 0.2:
#             patch.set_facecolor(colors_list[0])
#         elif bin_centers[i] < 0.4:
#             patch.set_facecolor(colors_list[1])
#         elif bin_centers[i] < 0.6:
#             patch.set_facecolor(colors_list[2])
#         elif bin_centers[i] < 0.8:
#             patch.set_facecolor(colors_list[3])
#         else:
#             patch.set_facecolor(colors_list[4])

#     # Styling for histogram
#     axes[1, 1].set_title('Distribution of Mean Similarities',
#                         fontsize=14, fontweight='bold', pad=20)
#     axes[1, 1].set_xlabel('Similarity', fontsize=12)
#     axes[1, 1].set_ylabel('Frequency', fontsize=12)
#     axes[1, 1].axvline(x=jnp.mean(all_similarities), color='#E74C3C', linestyle='--',
#                     linewidth=2, alpha=0.9,
#                     label=f'Mean: {jnp.mean(all_similarities):.2f}')
#     axes[1, 1].axvline(x=jnp.median(all_similarities), color='#8E44AD', linestyle=':',
#                     linewidth=2, alpha=0.9,
#                     label=f'Median: {jnp.median(all_similarities):.2f}')
#     axes[1, 1].legend(fontsize=10)
#     axes[1, 1].grid(True, alpha=0.2, linestyle='--', color='gray')
#     axes[1, 1].tick_params(axis='both', which='major', labelsize=10)

#     # Overall styling improvements
#     plt.suptitle(f'Policy Feature Similarity Analysis - {env_name}',
#                 fontsize=16, fontweight='bold', y=0.98)

#     plt.tight_layout()
#     plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300, facecolor='white')
#     plt.close()

#     print(f"Enhanced similarity heatmap saved to: {save_path}")

# # Create the visualization
# heatmap_save_path = save_path + f"similarity_heatmap_{env_name}.pdf"
# plot_similarity_heatmap(mean_similarity_matrix, sem_similarity_matrix, heatmap_save_path,
#                     title=f"Policy Feature Similarity Matrix - {env_name}")

# # Print some summary statistics
# print(f"\nSimilarity Analysis Summary for {env_name}:")
# print(f"Number of policies: {mean_similarity_matrix.shape[0]}")
# print(f"Number of observations: {n_observations}")
# print(f"Mean similarity (all pairs): {jnp.mean(mean_similarity_matrix):.2f} ± {jnp.mean(sem_similarity_matrix):.2f} SEM")
# print(f"Max similarity: {jnp.max(mean_similarity_matrix):.2f}")
# print(f"Min similarity: {jnp.min(mean_similarity_matrix):.2f}")
# print(f"Self-similarities (diagonal): {jnp.mean(jnp.diag(mean_similarity_matrix)):.2f} ± {jnp.mean(jnp.diag(sem_similarity_matrix)):.2f} SEM")

# print("Similarity matrix: ", similarity_matrices.shape)
# print("Single similarity matrix: ", similarity_matrices[0, :])
# print("Second similarity matrix: ", similarity_matrices[1, :])
# print("Most important features: ", most_important_features)
# print("Saliencies: ", saliencies.shape)
# print(saliencies[:, 1, :])
# print("Similarity matrices per observation: ", similarity_matrices.shape)
# print("Overall mean similarity: ", jnp.mean(similarity_matrices, axis=(0)))
# def top_feature_analysis(parameters): 
#     """
#     Analyze the top feature of the saliency map.
#     parameters: (num_policies, architecture)

#     Returns: (num_policies, num_policies) -> The similarity between the top features of the saliency map.
#     """
#     saliency_map_fn = build_saliency_map_fn(env_name)
#     multiple_params_saliency_map_fn = jax.vmap(saliency_map_fn, in_axes=(0, None))
#     params, saliencies = jax.lax.scan(
#         multiple_params_saliency_map_fn,
#         parameters,
#         xs=observations
#     )