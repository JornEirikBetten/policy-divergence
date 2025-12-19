import pandas as pd 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.ticker import MaxNLocator
import pickle
import jax.numpy as jnp
path_to_importance_agreement_data = os.getcwd() + "/interpretation_data/"
importance_agreement_data = pickle.load(open(path_to_importance_agreement_data + "importance_agreement_data.pkl", "rb"))

print(importance_agreement_data.keys())
envs = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
env_tags = ["Asterix", "Breakout", "Freeway", "Seaquest", "Space Invaders"]
training_setups = ["DWDI", "SWDI", "DWSI"]
training_labels = ["DWDI", "SWDI", "DWSI"]

def create_feature_overlap_histogram(data_dict, save_path="feature_overlap_histogram.pdf"):
    """
    Create an aesthetically pleasing histogram showing the distribution of feature overlap
    across all policy pairs (upper triangular only) for each environment.
    
    Args:
        similarities: Dictionary with keys 'environment', 'mean_similarity', 'sem_similarity'
        save_path: Path to save the plot
    """
    similarities_data = pd.DataFrame(data_dict)
    print(similarities_data.head())
    environments = envs
    n_envs = len(environments)
    n_training_setups = len(training_setups)
    training_labels = ["B", "AD", "ED"]
    fig, axs = plt.subplots(1, n_envs, figsize=(15, 4),
                            gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    # Set background
    fig.patch.set_facecolor('white')
    # Enhanced color palette matching the environment colors
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, (env, env_tag) in enumerate(zip(environments, env_tags)):
        env_data = similarities_data[similarities_data["environment"] == env]
        ax = axs[i]
        violin_data = []
        for j, training_setup in enumerate(training_setups):
            importance_agreement = env_data[env_data["training_setup"] == training_setup]["importance_agreement"].values
            importance_agreement = importance_agreement[0]
            violin_data.append(importance_agreement)
        # Prepare data for violin plot
        #violin_data = [np.array(importance_agreement)]
        
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
        ax.set_xlabel('training context', fontsize=12)#, fontweight='semibold')
        if i == 0:
            ax.set_ylabel('agreement in important states', fontsize=12)#, fontweight='semibold')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{env_tag}',
                        fontsize=16, fontweight='semibold', pad=10,
                        fontfamily='sans-serif', color='#34495E')
        ax.set_xticks(range(n_training_setups))
        ax.set_xticklabels(training_labels, fontsize=12)#, fontweight='semibold')
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y', color='gray')
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)
        ax.set_ylim(0, 1.0)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
    
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(save_path, format="pdf", bbox_inches="tight",
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Feature overlap histogram saved to: {save_path}")
    
create_feature_overlap_histogram(importance_agreement_data, save_path=path_to_importance_agreement_data + "overlap_in_most_important_states.pdf")

from statistical_tests import print_comparison_report
for env in envs:
    similarities_data = pd.DataFrame(importance_agreement_data)
    print(similarities_data.head())
    env_data = similarities_data[similarities_data["environment"] == env]
    print(env_data.head())
    imp_agreements = {}
    for training_setup in training_setups:
        importance_agreement = env_data[env_data["training_setup"] == training_setup]["importance_agreement"].values[0]
        imp_agreements[training_setup] = importance_agreement
    #print(f"Environment: {env}, Training setup: {training_setup}")
    print(f"Environment: {env}")
    print_comparison_report(imp_agreements["DWDI"], imp_agreements["SWDI"], label1="DWDI", label2="SWDI")
    print_comparison_report(imp_agreements["DWDI"], imp_agreements["DWSI"], label1="DWDI", label2="DWSI")
    print_comparison_report(imp_agreements["SWDI"], imp_agreements["DWSI"], label1="SWDI", label2="DWSI")