import jax 
import jax.numpy as jnp 
import haiku as hk 
import pgx 
import chex 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import sys 
import pickle 
from src.simulator.simulator import TwoSetTrajectory


env_name = sys.argv[1]
trajectory_path = os.getcwd() + "/trajectories/" + env_name + "/"
trajectory_path = trajectory_path + "same-world-trajectories.pkl"
with open(trajectory_path, "rb") as f:
    trajectories = pickle.load(f)



class TrajectoryAnalyzer: 
    
    def __init__(self, trajectories): 
        self.trajectories = trajectories
        print("Action distributions: ", self.trajectories.first_action_distribution.shape)
        print("Second action distributions: ", self.trajectories.second_action_distribution.shape)
        print("Actions: ", self.trajectories.action.shape)
        print("Accumulated rewards: ", self.trajectories.accumulated_rewards.shape)
        print("Randomness: ", self.trajectories.randomness.shape)
        
        print("State: ", self.trajectories.state.observation.shape)
        print("Action: ", self.trajectories.action.shape)
        
        self.conc_same_world_action_distributions = jnp.concatenate(self.trajectories.first_action_distribution, axis=0)
        self.conc_different_world_action_distributions = jnp.concatenate(self.trajectories.second_action_distribution, axis=0)
        self.conc_actions = jnp.concatenate(self.trajectories.action, axis=0)
        self.conc_randomness = jnp.concatenate(self.trajectories.randomness, axis=0)
        self.conc_states = jnp.concatenate(self.trajectories.state.observation, axis=0)
        self.conc_action = jnp.concatenate(self.trajectories.action, axis=0)
        self.same_world_action_entropies, self.different_world_action_entropies = self.calculate_state_importance()
        
        self.sorted_indices_same_world_action_entropies = self.sort_states_by_importance(self.same_world_action_entropies)
        self.sorted_indices_different_world_action_entropies = self.sort_states_by_importance(self.different_world_action_entropies)
        # Extract 10% most important states by highest-performing policy
        self.top_10_percent_same_world_action_entropies = self.sorted_indices_same_world_action_entropies[:int(self.same_world_action_entropies.shape[0] * 0.1), -1]
        self.top_10_percent_different_world_action_entropies = self.sorted_indices_different_world_action_entropies[:int(self.different_world_action_entropies.shape[0] * 0.1), -1]
        same_world_action_deviances = self.calculate_action_deviance(self.conc_same_world_action_distributions, self.conc_action)
        different_world_action_deviances = self.calculate_action_deviance(self.conc_different_world_action_distributions, self.conc_action)
        print("Same world action deviances: ", same_world_action_deviances.shape)
        print("Different world action deviances: ", different_world_action_deviances.shape)
        top_10_percent_same_world_action_deviances = same_world_action_deviances[self.top_10_percent_same_world_action_entropies]
        top_10_percent_different_world_action_deviances = different_world_action_deviances[self.top_10_percent_different_world_action_entropies]
        print("Top 10 percent same world action deviances: ", jnp.mean(top_10_percent_same_world_action_deviances))
        print("Same world action deviances: ", jnp.mean(same_world_action_deviances))
        print("Top 10 percent different world action deviances: ", jnp.mean(top_10_percent_different_world_action_deviances))
        print("Different world action deviances: ", jnp.mean(different_world_action_deviances))
        
    def calculate_action_deviance(self, action_distributions, actions):
        """
        action_distributions: (num_states, num_policies, num_actions)
        actions: (num_states, )
        Returns: (num_states, )
        """
        all_actions = jnp.argmax(action_distributions, axis=-1)
        check_deviance = lambda act, chosen_act: (act != chosen_act).astype(jnp.float32)
        deviances = jax.vmap(check_deviance, in_axes=(0, 0))(all_actions, actions)
        
        return deviances
        
    def calculate_state_importance(self):
        entropy = lambda x: -jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0)) 
        same_world_action_entropies = jnp.apply_along_axis(entropy, axis=-1, arr=self.conc_same_world_action_distributions)
        different_world_action_entropies = jnp.apply_along_axis(entropy, axis=-1, arr=self.conc_different_world_action_distributions)
        print("Same world action entropies: ", same_world_action_entropies.shape)
        print("Different world action entropies: ", different_world_action_entropies.shape)
        return same_world_action_entropies, different_world_action_entropies
    
    def hellinger_distance(self, p, q): 
        return jnp.sqrt(jnp.sum((jnp.sqrt(p) - jnp.sqrt(q))**2))
    
    def hellinger_kernel(self, Ps, Qs): 
        """
        Ps: (N, K)
        Qs: (N, K)
        Returns: (N, N)
        """
        triu_indices = jnp.triu_indices(Ps.shape[1], k=1)
        kernel = lambda P, Q: jax.vmap(jax.vmap(self.hellinger_distance, in_axes=(None, 0)), in_axes=(0, None))(P, Q)
        distances = jax.vmap(kernel, in_axes=(0, 0))(Ps, Qs)
        chosen_distances = jax.vmap(lambda x: x[triu_indices], in_axes=(0))(distances)
        mean_distances = jnp.mean(chosen_distances, axis=(1))
        return mean_distances
        
        
    def plot_state_importance(self):
        limits = (0, jnp.log(self.conc_same_world_action_distributions.shape[-1]))
        plt.style.use("seaborn-v0_8-darkgrid")
        n_policies = self.same_world_action_entropies.shape[1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Plot same world entropies
        axes[0].hist(
            self.same_world_action_entropies,
            bins=100,
            alpha=0.6,
            label=[f"Policy {i}" for i in range(n_policies)],
            color=["blue"]*n_policies,
        )
        axes[0].set_xlim(limits)
        axes[0].set_xlabel("Action Entropies")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Same World Entropies")
        # Only show legend if there are few enough policies
        if n_policies <= 5:
            axes[0].legend()

        # Plot different world entropies
        axes[1].hist(
            self.different_world_action_entropies,
            bins=100,
            alpha=0.6,
            label=[f"Policy {i}" for i in range(n_policies)],
            color=["red"]*n_policies,
        )
        axes[1].set_xlim(limits)
        axes[1].set_xlabel("Action Entropies")
        axes[1].set_title("Different World Entropies")
        if n_policies <= 5:
            axes[1].legend()

        plt.tight_layout()
        plt.savefig("state_importance.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        return 0
    
    def sort_states_by_importance(self, action_entropies):
        """
        Sort states by action entropies
        action_entropies: (num_states, num_policies, )
        Returns: (num_states, num_policies, )
        """
        print("Action entropies: ", action_entropies.shape)
        sorted_indices = jnp.argsort(action_entropies, axis=0)
        print("Sorted indices: ", sorted_indices.shape)
        return sorted_indices
    
        
analyzer = TrajectoryAnalyzer(trajectories)
spread_same_world = analyzer.hellinger_kernel(analyzer.conc_same_world_action_distributions, analyzer.conc_same_world_action_distributions)
spread_different_world = analyzer.hellinger_kernel(analyzer.conc_different_world_action_distributions, analyzer.conc_different_world_action_distributions)

top_10_percent_same_world_important_states_indices = analyzer.top_10_percent_same_world_action_entropies
top_10_percent_different_world_important_states_indices = analyzer.top_10_percent_different_world_action_entropies

same_world_act_dist_high_importance = analyzer.conc_same_world_action_distributions[top_10_percent_same_world_important_states_indices]
different_world_act_dist_high_importance = analyzer.conc_different_world_action_distributions[top_10_percent_different_world_important_states_indices]

spread_same_world_high_importance = analyzer.hellinger_kernel(same_world_act_dist_high_importance, same_world_act_dist_high_importance)
spread_different_world_high_importance = analyzer.hellinger_kernel(different_world_act_dist_high_importance, different_world_act_dist_high_importance)

print("Mean spread same world high importance: ", jnp.mean(spread_same_world_high_importance))
print("Mean spread different world high importance: ", jnp.mean(spread_different_world_high_importance))


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].hist(spread_same_world_high_importance, bins=100, alpha=0.5)
axs[0].set_title("Same world")
axs[1].hist(spread_different_world_high_importance, bins=100, alpha=0.5)
axs[1].set_title("Different world")
axs[0].set_xlabel("Spread")
axs[0].set_ylabel("Count")
axs[1].set_xlabel("Spread")
axs[1].set_ylabel("Count")
plt.savefig("spread_same_world_vs_spread_different_world.pdf", format="pdf", bbox_inches="tight")

#analyzer.plot_state_importance()