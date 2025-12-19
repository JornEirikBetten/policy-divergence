import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import colorsys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import pandas as pd
def load_similarity_data(pkl_path):
    """Load similarity data from pickle file"""
    with open(pkl_path, "rb") as f:
        similarities = pickle.load(f)
    return similarities

def create_enhanced_similarity_plot(similarities, save_path="similarity_comparison.png"):
    """
    Create a comprehensive plot comparing similarity matrices across all environments

    Args:
        similarities: Dictionary with keys 'environment', 'mean_similarity', 'sem_similarity'
        save_path: Path to save the plot
    """

    # Set up a more aesthetic style with seaborn
    plt.style.use('default')
    sns.set_palette("husl")

    # Define environment names and enhanced colors with better contrast
    environments = similarities["environment"]
    n_envs = len(environments)

    # Enhanced color palette for better visual distinction
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A90E2']  # Teal, Magenta, Orange, Red, Blue
    custom_cmaps = []

    for i, color in enumerate(colors):
        # Create sophisticated colormaps with better gradients
        if i % 2 == 0:
            # For even indices: white to color
            cmap = LinearSegmentedColormap.from_list(f"custom_{i}", ['#f8f9fa', color])
        else:
            # For odd indices: light color to dark color
            # Create lighter version by blending with white
            r, g, b = int(color[1:3], 16) / 255.0, int(color[3:5], 16) / 255.0, int(color[5:7], 16) / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            light_color = colorsys.hls_to_rgb(h, 0.9, s)  # Increase lightness
            light_hex = '#{:02x}{:02x}{:02x}'.format(int(light_color[0]*255), int(light_color[1]*255), int(light_color[2]*255))
            cmap = LinearSegmentedColormap.from_list(f"custom_{i}", [light_hex, color])
        custom_cmaps.append(cmap)

    # Create figure with subplots - improved layout
    if n_envs <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18),
                                gridspec_kw={'hspace': 0.35, 'wspace': 0.3})
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 2, figsize=(20, 22),
                                gridspec_kw={'hspace': 0.35, 'wspace': 0.3})
        axes = axes.flatten()

    # Add main title with better typography
    fig.suptitle('Policy Feature Similarity Comparison Across Environments',
                fontsize=24, fontweight='bold', y=0.995,
                fontfamily='sans-serif', color='#1a1a1a', alpha=0.9)

    # Add a subtle background color with gradient effect
    fig.patch.set_facecolor('white')

    # Plot each environment's similarity matrix
    for i, (env_name, mean_matrix, sem_matrix) in enumerate(zip(
        environments,
        similarities["mean_similarity"],
        similarities["sem_similarity"]
    )):

        ax = axes[i]
        n_policies = mean_matrix.shape[0]

        # Use custom colormap for this environment
        cmap = custom_cmaps[i % len(custom_cmaps)]

        # Main heatmap with cell values
        im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1, alpha=0.9)

        # Add cell values as text (smaller font for multiple plots)
        for j in range(n_policies):
            for k in range(n_policies):
                value = f'{mean_matrix[j, k]:.2f}'
                ax.text(k, j, value,
                       ha='center', va='center',
                       color='white' if mean_matrix[j, k] > 0.5 else 'black',
                       fontweight='bold',
                       fontsize=4)  # Very small font for multiple subplots

        # Enhanced styling for each subplot
        ax.set_title(f'{env_name}\n(n={similarities["n_observations"][i]} observations)',
                    fontsize=16, fontweight='bold', pad=20,
                    fontfamily='serif', color='#34495E')
        ax.set_xlabel('Policy Index', fontsize=12, fontweight='semibold')
        ax.set_ylabel('Policy Index', fontsize=12, fontweight='semibold')
        ax.set_xticks(range(n_policies))
        ax.set_yticks(range(n_policies))
        ax.tick_params(axis='both', which='major', labelsize=6, width=1)  # Very small font for all integer ticks
        ax.grid(True, alpha=0.1, linestyle='--', color='gray')

        # Add subtle border around each heatmap
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)

        # Add colorbar for each subplot with enhanced styling
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
        cbar.ax.tick_params(labelsize=9, width=2)
        cbar.set_label('Similarity', rotation=270, labelpad=15, fontsize=11, fontweight='semibold')
        cbar.outline.set_edgecolor('#BDC3C7')
        cbar.outline.set_linewidth(1.5)

    # Add summary statistics subplot
    if n_envs < len(axes):
        ax_summary = axes[-1]

        # Calculate summary statistics for each environment
        env_means = [jnp.mean(mean_matrix) for mean_matrix in similarities["mean_similarity"]]
        env_stds = [jnp.std(mean_matrix) for mean_matrix in similarities["mean_similarity"]]
        env_self_similarities = [jnp.mean(jnp.diag(mean_matrix)) for mean_matrix in similarities["mean_similarity"]]

        # Create bar plot
        x_pos = np.arange(n_envs)
        width = 0.25

        bars1 = ax_summary.bar(x_pos - width, env_means, width,
                              label='Mean Similarity', alpha=0.8, color=colors[0])
        bars2 = ax_summary.bar(x_pos, env_self_similarities, width,
                              label='Self-Similarity', alpha=0.8, color=colors[1])
        bars3 = ax_summary.bar(x_pos + width, env_stds, width,
                              label='Std Deviation', alpha=0.8, color=colors[2])

        # Enhanced styling for summary plot
        ax_summary.set_title('Summary Statistics Across Environments',
                            fontsize=18, fontweight='bold', pad=20,
                            fontfamily='serif', color='#34495E')
        ax_summary.set_xlabel('Environment', fontsize=13, fontweight='semibold')
        ax_summary.set_ylabel('Similarity Value', fontsize=13, fontweight='semibold')
        ax_summary.set_xticks(x_pos)
        ax_summary.set_xticklabels(environments, rotation=45, ha='right', fontsize=11, fontweight='semibold')
        ax_summary.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax_summary.grid(True, alpha=0.2, axis='y', linestyle='--', color='gray')

        # Enhanced tick styling
        ax_summary.tick_params(axis='both', which='major', labelsize=10, width=2)
        ax_summary.spines['top'].set_visible(False)
        ax_summary.spines['right'].set_visible(False)
        ax_summary.spines['left'].set_linewidth(1.5)
        ax_summary.spines['bottom'].set_linewidth(1.5)

        # Add value labels on bars with enhanced styling
        def add_value_labels(ax, bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

        add_value_labels(ax_summary, bars1)
        add_value_labels(ax_summary, bars2)
        add_value_labels(ax_summary, bars3)

    # Hide empty subplots if any
    for i in range(n_envs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save the plot with enhanced styling
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=350,
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

    print(f"Enhanced similarity comparison plot saved to: {save_path}")

    return fig

def create_heatmap_grid(similarities, save_path="similarity_heatmap_grid.png"):
    """
    Create a grid of heatmaps showing all similarity matrices side by side
    """

    environments = similarities["environment"]
    n_envs = len(environments)

    # Create figure with subplots in a grid - improved layout
    cols = min(3, n_envs)
    rows = (n_envs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows),
                            gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Enhanced color palette for grid layout
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A90E2']
    custom_cmaps = []

    for color in colors:
        # Create sophisticated colormaps for grid
        cmap = LinearSegmentedColormap.from_list(f"grid_{color}", ['#f8f9fa', color])
        custom_cmaps.append(cmap)

    for i, (env_name, mean_matrix) in enumerate(zip(environments, similarities["mean_similarity"])):

        ax = axes_flat[i]
        n_policies = mean_matrix.shape[0]

        # Use different colormap for each environment
        cmap = custom_cmaps[i % len(custom_cmaps)]

        # Create heatmap
        im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1, alpha=0.9)

        # Add cell values (very small font for grid layout)
        for j in range(n_policies):
            for k in range(n_policies):
                value = f'{mean_matrix[j, k]:.2f}'
                ax.text(k, j, value,
                       ha='center', va='center',
                       color='white' if mean_matrix[j, k] > 0.5 else 'black',
                       fontweight='bold',
                       fontsize=3)  # Extra small for grid

        # Enhanced styling for grid layout - add integer ticks
        ax.set_xticks(range(n_policies))
        ax.set_yticks(range(n_policies))
        ax.tick_params(axis='both', which='major', labelsize=4, width=1)  # Extra small font for grid
        ax.set_title(f'{env_name}', fontsize=14, fontweight='bold', pad=15,
                    fontfamily='serif', color='#34495E')

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#BDC3C7')
            spine.set_linewidth(1.5)

        # Add colorbar (only for first plot to avoid clutter)
        if i == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.08, shrink=0.8)
            cbar.set_label('Similarity', rotation=270, labelpad=15, fontsize=11, fontweight='semibold')
            cbar.outline.set_edgecolor('#BDC3C7')
            cbar.outline.set_linewidth(1.5)

    # Hide empty subplots
    for i in range(n_envs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Add main title with better typography
    fig.suptitle('Policy Feature Similarity Matrices - All Environments',
                fontsize=18, fontweight='bold', y=0.98,
                fontfamily='serif', color='#2C3E50')

    # Add subtle background
    fig.patch.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=350,
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

    print(f"Enhanced heatmap grid saved to: {save_path}")

def create_feature_overlap_histogram(similarities, save_path="feature_overlap_histogram.pdf"):
    """
    Create an aesthetically pleasing histogram showing the distribution of feature overlap
    across all policy pairs (upper triangular only) for each environment.
    
    Args:
        similarities: Dictionary with keys 'environment', 'mean_similarity', 'sem_similarity'
        save_path: Path to save the plot
    """
    similarities_data = pd.DataFrame(dict(similarities))
    print(similarities_data.head())
    environments = np.unique(similarities["environment"])
    training_setups = ["different-world", "same-world", "different-world-same-init"]
    n_envs = len(environments)
    n_training_setups = len(training_setups)
    training_labels = ["B", "AD", "ED"]
    fig, axs = plt.subplots(1, n_envs, figsize=(15, 4),
                            gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    triu_indices = jnp.triu_indices(20, k=1)
    # Set background
    fig.patch.set_facecolor('white')
    # Enhanced color palette matching the environment colors
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A90E2']
    all_violin_data = {
        "environment": [],
        "training_setup": [],
        "overlaps": []
    }
    for i, env in enumerate(environments):
        env_data = similarities_data[similarities_data["environment"] == env]
        ax = axs[i]
        violin_data = []
        upper_tri_values = []
        for j, training_setup in enumerate(training_setups):
            #similarity_data = similarities[similarities["environment"] == env and similarities["training_setup"] == training_setup]
            
            mean_similarity = env_data[env_data["training_setup"] == training_setup]["mean_similarity"].values
            mean_similarity = mean_similarity[0]
            #print(mean_similarity.shape)
            values = mean_similarity[triu_indices]
            all_violin_data["environment"].append(env)
            all_violin_data["training_setup"].append(training_setup)
            all_violin_data["overlaps"].append(values)
            #print(values)
            upper_tri_values.append(values)
            #print(f"Environment: {env}, Training setup: {training_labels[j]}, Values: {mean_similarity}")
            # sem_similarity = similarities["sem_similarity"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
            # n_observations = similarities["n_observations"][similarities["environment"] == env and similarities["training_setup"] == training_setup]
            # print(f"Environment: {env}, Training setup: {training_setup}, Mean similarity: {mean_similarity}, SEM similarity: {sem_similarity}, N observations: {n_observations}")
        # Prepare data for violin plot
        violin_data = [np.array(values) for values in upper_tri_values]
        ax.set_xlim(-1, 3)
        # Create violin plot
        parts = ax.violinplot(violin_data, positions=range(n_training_setups),
                                        showmeans=True, widths=0.5)#, showmedians=True, widths=0.7)
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
        ax.set_xlabel('training setup', fontsize=12)#, fontweight='semibold')
        if i == 0:
            ax.set_ylabel('top 5 features in common', fontsize=12)#, fontweight='semibold')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{env}',
                        fontsize=16, fontweight='semibold', pad=10,
                        fontfamily='sans-serif', color='#34495E')
        ax.set_xticks(range(n_training_setups))
        ax.set_xticklabels(training_labels, fontsize=12)#, fontweight='semibold')
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)
        ax.set_ylim(0, 1)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.set_xlabel("training context", fontsize=12)
    # # Enhanced color palette matching the environment colors
    # colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4A90E2']
    
    # # Extract upper triangular values for each environment
    # upper_tri_values = []
    # for mean_matrix in similarities["mean_similarity"]:
    #     # Get indices of upper triangle (excluding diagonal)
    #     triu_indices = jnp.triu_indices(mean_matrix.shape[0], k=1)
    #     values = mean_matrix[triu_indices]
    #     upper_tri_values.append(values)
    
    # # Create figure
    # fig, axes = plt.subplots(2, 1, figsize=(16, 12),
    #                         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
    
    # # Set background
    # fig.patch.set_facecolor('white')
    
    # # Main title
    # fig.suptitle('Distribution of Feature Overlap Between Policy Pairs',
    #             fontsize=24, fontweight='bold', y=0.98,
    #             fontfamily='sans-serif', color='#1a1a1a', alpha=0.9)
    
    # # Top panel: Overlapping histograms
    # ax_main = axes[0]
    
    # # Plot histograms for each environment
    # for i, (env_name, values) in enumerate(zip(environments, upper_tri_values)):
    #     ax_main.hist(values, bins=25, alpha=0.6, label=env_name,
    #                 color=colors[i], edgecolor='black', linewidth=1.2)
    
    # # Styling for main histogram
    # ax_main.set_xlabel('Feature Similarity', fontsize=16, fontweight='semibold')
    # ax_main.set_ylabel('Frequency', fontsize=16, fontweight='semibold')
    # ax_main.set_title('Overlapping Distributions of Policy Pair Similarities',
    #                  fontsize=18, fontweight='bold', pad=20,
    #                  fontfamily='sans-serif', color='#34495E')
    # ax_main.legend(fontsize=13, loc='upper right', framealpha=0.95,
    #               edgecolor='#BDC3C7', fancybox=True, shadow=True)
    # ax_main.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    # ax_main.tick_params(axis='both', which='major', labelsize=12, width=2)
    # ax_main.set_xlim(0, 1)
    
    # # Remove top and right spines
    # ax_main.spines['top'].set_visible(False)
    # ax_main.spines['right'].set_visible(False)
    # ax_main.spines['left'].set_linewidth(2)
    # ax_main.spines['bottom'].set_linewidth(2)
    
    # # Bottom panel: Violin plot showing distributions
    # ax_violin = axes[1]
    
    # # Prepare data for violin plot
    # violin_data = [np.array(values) for values in upper_tri_values]
    
    # # Create violin plot
    # parts = ax_violin.violinplot(violin_data, positions=range(len(environments)),
    #                              showmeans=True, showmedians=True, widths=0.7)
    
    # # Color the violin plots
    # for i, pc in enumerate(parts['bodies']):
    #     pc.set_facecolor(colors[i])
    #     pc.set_alpha(0.7)
    #     pc.set_edgecolor('black')
    #     pc.set_linewidth(1.5)
    
    # # Style the violin plot components
    # for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans']:
    #     if partname in parts:
    #         parts[partname].set_edgecolor('black')
    #         parts[partname].set_linewidth(2)
    
    # # Styling for violin plot
    # ax_violin.set_xlabel('environment', fontsize=16, fontweight='semibold')
    # ax_violin.set_ylabel('overlapping feature attribution', fontsize=16, fontweight='semibold')
    # ax_violin.set_title('Distribution Summary by Environment (Violin Plot)',
    #                    fontsize=18, fontweight='bold', pad=20,
    #                    fontfamily='sans-serif', color='#34495E')
    # ax_violin.set_xticks(range(len(environments)))
    # ax_violin.set_xticklabels(environments, fontsize=12, fontweight='semibold')
    # ax_violin.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
    # ax_violin.tick_params(axis='both', which='major', labelsize=12, width=2)
    # ax_violin.set_ylim(0, 1)
    
    # # Remove top and right spines
    # ax_violin.spines['top'].set_visible(False)
    # ax_violin.spines['right'].set_visible(False)
    # ax_violin.spines['left'].set_linewidth(2)
    # ax_violin.spines['bottom'].set_linewidth(2)
    
    # Add statistical annotations
    # for i, (env_name, values) in enumerate(zip(environments, upper_tri_values)):
    #     mean_val = jnp.mean(values)
    #     median_val = jnp.median(values)
    #     ax_violin.text(i, 0.95, f'μ={mean_val:.2f}\nM={median_val:.2f}',
    #                   ha='center', va='top', fontsize=9,
    #                   bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[i],
    #                            alpha=0.3, edgecolor='black', linewidth=1))
    
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(save_path, format="pdf", bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close()
    data_save_path = save_path.replace(".pdf", "_data.pkl")
    with open(data_save_path, "wb") as f:
        pickle.dump(all_violin_data, f)
    print(f"Feature overlap histogram saved to: {save_path}")
    
    # Print statistics
    print("\nFeature Overlap Statistics (Upper Triangular Only):")
    for i, (env_name, values) in enumerate(zip(environments, upper_tri_values)):
        print(f"\n{env_name}:")
        print(f"  Number of policy pairs: {len(values)}")
        print(f"  Mean similarity: {jnp.mean(values):.3f}")
        print(f"  Median similarity: {jnp.median(values):.3f}")
        print(f"  Std deviation: {jnp.std(values):.3f}")
        print(f"  Min similarity: {jnp.min(values):.3f}")
        print(f"  Max similarity: {jnp.max(values):.3f}")

if __name__ == "__main__":
    # Path to the similarities pickle file
    similarities_path = "/home/jeb/Projects/variation-in-rl/feature_analysis/similarities.pkl"

    # Load the data
    similarities = load_similarity_data(similarities_path)

    # Create comprehensive comparison plot
    save_path_1 = "/home/jeb/Projects/variation-in-rl/feature_analysis/similarity_comparison_comprehensive.png"
    #create_enhanced_similarity_plot(similarities, save_path_1)

    # Create heatmap grid
    save_path_2 = "/home/jeb/Projects/variation-in-rl/feature_analysis/similarity_heatmap_grid.png"
    #create_heatmap_grid(similarities, save_path_2)

    # Create feature overlap histogram
    save_path_3 = "/home/jeb/Projects/variation-in-rl/feature_analysis/feature_overlap.pdf"
    create_feature_overlap_histogram(similarities, save_path_3)

    print(f"\nSimilarity Analysis Summary:")
    print(f"Number of environments: {len(similarities['environment'])}")
    print(f"Environments: {', '.join(similarities['environment'])}")

    # Calculate and display summary statistics
    for i, env_name in enumerate(similarities['environment']):
        mean_matrix = similarities['mean_similarity'][i]
        n_obs = similarities['n_observations'][i]

        print(f"\n{env_name} (n={n_obs}):")
        print(f"  Mean similarity: {jnp.mean(mean_matrix):.2f}")
        print(f"  Self-similarity: {jnp.mean(jnp.diag(mean_matrix)):.2f}")
        print(f"  Max similarity: {jnp.max(mean_matrix):.2f}")
        print(f"  Min similarity: {jnp.min(mean_matrix):.2f}")
