import jax
import jax.numpy as jnp
from src.simulator.simulator import Trajectory, TwoSetTrajectory, ThreeSetTrajectory
import pickle 
import os 
import sys 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 




class TrajectoryAnalyzer: 
    def __init__(self, trajectories: ThreeSetTrajectory, env_name: str): 
        self.trajectories = trajectories
        
        self.dwdi_action_distributions = jnp.concatenate(self.trajectories.dwdi_action_distribution, axis=0)
        self.swdi_action_distributions = jnp.concatenate(self.trajectories.swdi_action_distribution, axis=0)
        self.dwsi_action_distributions = jnp.concatenate(self.trajectories.dwsi_action_distribution, axis=0)
        self.num_actions = self.dwdi_action_distributions.shape[-1]
        
        self.dwdi_actions = jnp.argmax(self.dwdi_action_distributions, axis=-1)
        self.swdi_actions = jnp.argmax(self.swdi_action_distributions, axis=-1)
        self.dwsi_actions = jnp.argmax(self.dwsi_action_distributions, axis=-1)
        
        self.dwdi_majority_actions = self.find_majority_action(self.dwdi_actions)
        self.swdi_majority_actions = self.find_majority_action(self.swdi_actions)
        self.dwsi_majority_actions = self.find_majority_action(self.dwsi_actions)
        
        self.dwdi_action_deviances = jax.vmap(self.calculate_action_deviance, in_axes=(0, 0))(self.dwdi_actions, self.dwdi_majority_actions)
        self.swdi_action_deviances = jax.vmap(self.calculate_action_deviance, in_axes=(0, 0))(self.swdi_actions, self.swdi_majority_actions)
        self.dwsi_action_deviances = jax.vmap(self.calculate_action_deviance, in_axes=(0, 0))(self.dwsi_actions, self.dwsi_majority_actions)
        
        self.all_dwdi_state_importances = self.calculate_state_importance(self.dwdi_action_distributions)
        self.all_swdi_state_importances = self.calculate_state_importance(self.swdi_action_distributions)
        self.all_dwsi_state_importances = self.calculate_state_importance(self.dwsi_action_distributions)
        
        # print(self.all_swdi_state_importances[:100, 0])
        # print(self.all_swdi_state_importances[:100, 3])
        
        self.best_dwdi_state_importances = self.all_dwdi_state_importances[:, -1]
        self.best_swdi_state_importances = self.all_swdi_state_importances[:, -1]
        self.best_dwsi_state_importances = self.all_dwsi_state_importances[:, -1]
        
        self.most_important_dwdi_states = jnp.argsort(self.best_dwdi_state_importances)[:int(self.best_dwdi_state_importances.shape[0] * 0.1)]
        self.most_important_swdi_states = jnp.argsort(self.best_swdi_state_importances)[:int(self.best_swdi_state_importances.shape[0] * 0.1)]
        self.most_important_dwsi_states = jnp.argsort(self.best_dwsi_state_importances)[:int(self.best_dwsi_state_importances.shape[0] * 0.1)]
        
        self.action_deviances_in_most_important_dwdi_states = self.dwdi_action_deviances[self.most_important_dwdi_states]
        self.action_deviances_in_most_important_swdi_states = self.swdi_action_deviances[self.most_important_swdi_states]
        self.action_deviances_in_most_important_dwsi_states = self.dwsi_action_deviances[self.most_important_dwsi_states]
        
        self.dwdi_mean_action_deviance = jnp.mean(self.dwdi_action_deviances, axis=0)
        self.swdi_mean_action_deviance = jnp.mean(self.swdi_action_deviances, axis=0)
        self.dwsi_mean_action_deviance = jnp.mean(self.dwsi_action_deviances, axis=0)
        self.dwdi_stderr_action_deviance = jnp.std(self.dwdi_action_deviances, axis=0)/jnp.sqrt(self.dwdi_action_deviances.shape[0])
        self.swdi_stderr_action_deviance = jnp.std(self.swdi_action_deviances, axis=0)/jnp.sqrt(self.swdi_action_deviances.shape[0])
        self.dwsi_stderr_action_deviance = jnp.std(self.dwsi_action_deviances, axis=0)/jnp.sqrt(self.dwsi_action_deviances.shape[0])
        
        self.dwdi_mean_action_deviance_in_most_important_states = jnp.mean(self.action_deviances_in_most_important_dwdi_states, axis=0)
        self.swdi_mean_action_deviance_in_most_important_states = jnp.mean(self.action_deviances_in_most_important_swdi_states, axis=0)
        self.dwsi_mean_action_deviance_in_most_important_states = jnp.mean(self.action_deviances_in_most_important_dwsi_states, axis=0)
        self.dwdi_stderr_action_deviance_in_most_important_states = jnp.sqrt(self.dwdi_mean_action_deviance_in_most_important_states * (1 - self.dwdi_mean_action_deviance_in_most_important_states) / self.dwdi_action_deviances.shape[0])
        self.swdi_stderr_action_deviance_in_most_important_states = jnp.sqrt(self.swdi_mean_action_deviance_in_most_important_states * (1 - self.swdi_mean_action_deviance_in_most_important_states) / self.swdi_action_deviances.shape[0])
        self.dwsi_stderr_action_deviance_in_most_important_states = jnp.sqrt(self.dwsi_mean_action_deviance_in_most_important_states * (1 - self.dwsi_mean_action_deviance_in_most_important_states) / self.dwsi_action_deviances.shape[0])
        
        print("-"*50)
        print(f"Environment: {env_name}")
        print(f"Action deviances in all DWDI states: {jnp.mean(self.dwdi_mean_action_deviance):.2e}+-{jnp.mean(self.dwdi_stderr_action_deviance):.2e}")
        print(f"Action deviances in all SWDI states: {jnp.mean(self.swdi_mean_action_deviance):.2e}+-{jnp.mean(self.swdi_stderr_action_deviance):.2e}")
        print(f"Action deviances in all DWSI states: {jnp.mean(self.dwsi_mean_action_deviance):.2e}+-{jnp.mean(self.dwsi_stderr_action_deviance):.2e}")
        print(f"Action deviances in most important DWDI states: {jnp.mean(self.dwdi_mean_action_deviance_in_most_important_states):.2e}+-{jnp.mean(self.dwdi_stderr_action_deviance_in_most_important_states):.2e}")
        print(f"Action deviances in most important SWDI states: {jnp.mean(self.swdi_mean_action_deviance_in_most_important_states):.2e}+-{jnp.mean(self.swdi_stderr_action_deviance_in_most_important_states):.2e}")
        print(f"Action deviances in most important DWSI states: {jnp.mean(self.dwsi_mean_action_deviance_in_most_important_states):.2e}+-{jnp.mean(self.dwsi_stderr_action_deviance_in_most_important_states):.2e}")
        
        self.dwdi_importance_agreement, self.dwdi_highest_importance_states = self.calculate_importance_agreement(self.all_dwdi_state_importances)
        self.swdi_importance_agreement, self.swdi_highest_importance_states = self.calculate_importance_agreement(self.all_swdi_state_importances)
        self.dwsi_importance_agreement, self.dwsi_highest_importance_states = self.calculate_importance_agreement(self.all_dwsi_state_importances)
       
        
        self.num_policies = self.all_dwdi_state_importances.shape[-1]
        triu_indices = jnp.triu_indices(self.num_policies, k=1)
        self.dwdi_importance_agreement_ut = self.dwdi_importance_agreement[triu_indices]
        self.swdi_importance_agreement_ut = self.swdi_importance_agreement[triu_indices]
        self.dwsi_importance_agreement_ut = self.dwsi_importance_agreement[triu_indices]
        self.dwdi_importance_agreement_mean = jnp.mean(self.dwdi_importance_agreement)
        self.swdi_importance_agreement_mean = jnp.mean(self.swdi_importance_agreement)
        self.dwsi_importance_agreement_mean = jnp.mean(self.dwsi_importance_agreement)
        self.dwdi_importance_agreement_stderr = jnp.std(self.dwdi_importance_agreement)/jnp.sqrt(self.dwdi_importance_agreement.shape[0])
        self.swdi_importance_agreement_stderr = jnp.std(self.swdi_importance_agreement)/jnp.sqrt(self.swdi_importance_agreement.shape[0])
        self.dwsi_importance_agreement_stderr = jnp.std(self.dwsi_importance_agreement)/jnp.sqrt(self.dwsi_importance_agreement.shape[0])
        print(f"DWDI importance agreement: {self.dwdi_importance_agreement_mean:.2e}+-{self.dwdi_importance_agreement_stderr:.2e}")
        print(f"SWDI importance agreement: {self.swdi_importance_agreement_mean:.2e}+-{self.swdi_importance_agreement_stderr:.2e}")
        print(f"DWSI importance agreement: {self.dwsi_importance_agreement_mean:.2e}+-{self.dwsi_importance_agreement_stderr:.2e}")
        print("-"*50)
        
        # Agreement between policies in most important states
        
    def calculate_action_deviance(self, actions, majority_actions): 
        """
        actions: (num_states, num_policies)
        majority_actions: (num_states, )
        Returns: (num_states, num_policies)
        """
        deviances = (actions != majority_actions).astype(jnp.float32)
        return deviances
    
    def find_majority_action(self, actions):
        """
        actions: (num_states, num_policies)
        Returns: (num_states, )
        """
        counts = jnp.apply_along_axis(lambda x: jnp.bincount(x, length=self.num_actions), axis=1, arr=actions)
        majority_actions = jnp.argmax(counts, axis=-1)
        return majority_actions
    
    def calculate_state_importance(self, action_distributions):
        """
        action_distributions: (num_states, num_policies, num_actions)
        Returns: (num_states, num_policies)
        """
        entropy = lambda x: -jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0))
        action_entropies = jnp.apply_along_axis(entropy, axis=-1, arr=action_distributions)
        return action_entropies
    
    def calculate_importance_agreement(self, state_importances): 
        """
        state_importances: (num_states, num_policies)
        Returns: (num_states, )
        """
        #state_importances = jnp.swapaxes(state_importances, 0, 1)
        # Extract a subset of the highest important states for each policy 
        extract_highest_important_states = lambda x: jnp.argsort(x)[:int(x.shape[0] * 0.1)]
        highest_importance_states = jax.vmap(extract_highest_important_states, in_axes=(1))(state_importances)
        agreement = jax.vmap(jax.vmap(self.overlap_kernel, in_axes=(0, None)), in_axes=(None, 0))(highest_importance_states, highest_importance_states)
        agreement = jnp.sum(agreement, axis=-1)/(highest_importance_states.shape[-1])
        return agreement, highest_importance_states
        
    def overlap_kernel(self, set1, set2): 
        """
        set1: (num_states, )
        set2: (num_states, )
        Returns: ()
        """
        overlap = jax.vmap(jnp.isin, in_axes=(0, None))(set1, set2)
        return overlap


def plot_histogram(data, ax): 
    ax.hist(data, bins=10, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xlabel("state importance")
    #ax.set_ylabel("frequency")
    return ax 

save_path = os.getcwd() + "/interpretation_data/"
os.makedirs(save_path, exist_ok=True)

env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]#["minatar-breakout"] #
action_deviance_data = {
    "environment": [], 
    "training_setup": [], 
    "policy_id": [],
    "fraction_action_deviance": [], 
    "fraction_action_deviance_in_most_important_states": [], 
}
importance_agreement_data = {
    "environment": [], 
    "training_setup": [], 
    "importance_agreement": [], 
}
for env_name in env_names:
    path_to_trajectories = os.getcwd() + "/trajectories/" + env_name + "/"
    path_to_trajectories = path_to_trajectories + "same-world-trajectories.pkl"
    with open(path_to_trajectories, "rb") as f:
        trajectories = pickle.load(f)    
    analyst = TrajectoryAnalyzer(trajectories, env_name)
    for policy_id in range(1, analyst.num_policies+1, 1):
        action_deviance_data["environment"].append(env_name)
        action_deviance_data["policy_id"].append(policy_id)
        action_deviance_data["fraction_action_deviance"].append(analyst.dwdi_mean_action_deviance[policy_id-1])
        action_deviance_data["fraction_action_deviance_in_most_important_states"].append(analyst.dwdi_mean_action_deviance_in_most_important_states[policy_id-1])
        action_deviance_data["training_setup"].append("DWDI")
        action_deviance_data["environment"].append(env_name)
        action_deviance_data["policy_id"].append(policy_id)
        action_deviance_data["fraction_action_deviance"].append(analyst.swdi_mean_action_deviance[policy_id-1])
        action_deviance_data["fraction_action_deviance_in_most_important_states"].append(analyst.swdi_mean_action_deviance_in_most_important_states[policy_id-1])
        action_deviance_data["training_setup"].append("SWDI")
        action_deviance_data["environment"].append(env_name)
        action_deviance_data["policy_id"].append(policy_id)
        action_deviance_data["fraction_action_deviance"].append(analyst.dwsi_mean_action_deviance[policy_id-1])
        action_deviance_data["fraction_action_deviance_in_most_important_states"].append(analyst.dwsi_mean_action_deviance_in_most_important_states[policy_id-1])
        action_deviance_data["training_setup"].append("DWSI")
    
    importance_agreement_data["environment"].append(env_name)
    importance_agreement_data["importance_agreement"].append(analyst.dwdi_importance_agreement_ut)
    #importance_agreement_data["stderr_importance_agreement"].append(analyst.dwdi_importance_agreement)
    importance_agreement_data["training_setup"].append("DWDI")
    importance_agreement_data["environment"].append(env_name)
    importance_agreement_data["importance_agreement"].append(analyst.swdi_importance_agreement_ut)
    #importance_agreement_data["stderr_importance_agreement"].append(analyst.swdi_importance_agreement_stderr)
    importance_agreement_data["training_setup"].append("SWDI")
    importance_agreement_data["environment"].append(env_name)
    importance_agreement_data["importance_agreement"].append(analyst.dwsi_importance_agreement_ut)
    #importance_agreement_data["stderr_importance_agreement"].append(analyst.dwsi_importance_agreement_stderr)
    importance_agreement_data["training_setup"].append("DWSI")
    
    
    fig, axs = plt.subplots(3, 20, figsize=(20, 4))
    for i in range(20):
        plot_histogram(analyst.all_dwdi_state_importances[:, i], axs[0, i])
        plot_histogram(analyst.all_swdi_state_importances[:, i], axs[1, i])
        plot_histogram(analyst.all_dwsi_state_importances[:, i], axs[2, i])
    plt.savefig(os.getcwd() + f"/interpretation_data/{env_name}_state_importance_histogram.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    


df = pd.DataFrame(action_deviance_data)
df.to_csv(save_path + "action_deviance_data.csv", index=False)

pickle_path = save_path + "importance_agreement_data.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(importance_agreement_data, f)


