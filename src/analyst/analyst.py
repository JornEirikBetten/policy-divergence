import jax.numpy as jnp 
import jax 
import haiku as hk 
import chex 
import pgx 
from typing import NamedTuple, Callable, Tuple
from dataclasses import fields 

import matplotlib.pyplot as plt 


class Trajectory(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    action_distribution: chex.Array
    randomness: jax.random.PRNGKey


class Analyst: 
    
    def __init__(self, trajectories: Trajectory): 
        self.trajectories = trajectories
        self.action_entropies = self._calculate_action_entropies()
        print(trajectories.action_distribution.shape)
        self.observations = jnp.concatenate(trajectories.state.observation, axis=0)
        self.action_entropies = jnp.concatenate(self.action_entropies, axis=0)
        self.reshaped_actions = jnp.concatenate(trajectories.action, axis=0)
        self.reshaped_action_distributions = jnp.concatenate(trajectories.action_distribution, axis=0)
        # self.observations = trajectories.state.observation.reshape((-1,) + trajectories.state.observation.shape[2:])
        # self.action_entropies = self.action_entropies.reshape(((-1,) + self.action_entropies.shape[2:]))
        # self.reshaped_actions = trajectories.action.reshape((-1,) + trajectories.action.shape[2:])
        # self.reshaped_action_distributions = trajectories.action_distribution.reshape((-1,) + trajectories.action_distribution.shape[2:])
        #print("Reshaped action distributions: ", self.reshaped_action_distributions.shape)
        self.chosen_actions = jnp.argmax(self.reshaped_action_distributions, axis=-1)
        #print("Chosen actions: ", self.chosen_actions.shape)
        self.reshaped_states = jax.tree.map(self._reshape_states, trajectories.state)
        self.deviations = self._calculate_action_deviations()
        #print("Deviations: ", self.deviations.shape)
        self.statewise_deviations = jnp.mean(self.deviations, axis=0)
        print("Max entropy: ", jnp.max(self.action_entropies))
        print("Min entropy: ", jnp.min(self.action_entropies))
        #print(f"Percentage deviation from majority: {jnp.mean(self.percentage_action_deviations):.2f}+-{jnp.std(self.percentage_action_deviations)/jnp.sqrt(self.percentage_action_deviations.shape[0]):.2f}")
        
        self.top_important_states, top_important_indices, self.batched_states, self.batched_chosen_actions, self.top_1000_batched_states, self.top_1000_batched_chosen_actions, self.majority_voted_states, self.majority_voted_chosen_actions = self._get_most_important_states()
        self.top_important_action_distributions = self.reshaped_action_distributions[top_important_indices]
        self.top_important_chosen_actions = self.chosen_actions[top_important_indices]
        self.top_important_actions = self.reshaped_actions[top_important_indices]
        #self.all_deviations = jax.vmap(self._check_deviation, in_axes=(-1, None))(self.chosen_actions, self.reshaped_actions)
        high_importance_deviations = jax.vmap(self._check_deviation, in_axes=(-1, None))(self.top_important_chosen_actions, self.top_important_actions)
        self.high_importance_statewise_deviations = jnp.mean(high_importance_deviations, axis=0)
        print(f"Percentage deviation from majority in all states: {jnp.mean(self.statewise_deviations):.2f}+-{jnp.std(self.statewise_deviations)/jnp.sqrt(self.statewise_deviations.shape[0]):.2f}")
        print(f"Percentage deviation from majority in 10 percent most important states: {jnp.mean(self.high_importance_statewise_deviations):.2f}+-{jnp.std(self.high_importance_statewise_deviations)/jnp.sqrt(self.high_importance_statewise_deviations.shape[0]):.2f}")
        
        #self.top_1000_important_states = jax.tree_util.tree_map(lambda x: x[top_important_indices], trajectories.state)
        
    def _reshape_states(self, x): 
        return jnp.concatenate(x, axis=0)
        # if len(x.shape) == 3: 
        #     if x.shape[2] == 0: 
        #         return x.reshape((x.shape[0]*x.shape[1], 0))
        #     else: 
        #         return x.reshape((-1,) + x.shape[2:])
        # else: 
        #     return x.reshape((-1,) + x.shape[2:])
        
        
    def _calculate_action_entropies(self): 
        action_distributions = self.trajectories.action_distribution
        entropy = lambda x: -jnp.sum(jnp.where(x > 0, x * jnp.log(x), 0))
        action_entropies = jnp.apply_along_axis(entropy, axis=-1, arr=action_distributions)
        return action_entropies 
    
    def _calculate_action_deviations(self): 
        policy_actions = self.chosen_actions
        majority_actions = self.reshaped_actions
        # print("Policy actions: ", policy_actions.shape)
        # print("Majority actions: ", majority_actions.shape)
        deviations = jax.vmap(self._check_deviation, in_axes=(-1, None))(policy_actions, majority_actions)
        return deviations
    
    def _check_deviation(self, chosen_actions, majority_actions): 
        deviation = (chosen_actions != majority_actions).astype(jnp.float32)
        return deviation
    
    # TODO: Get most important states 
    def _get_most_important_states(self): 
        action_entropies_reference = self.action_entropies[:, -1]
        mean_action_distributions = jnp.mean(self.reshaped_action_distributions, axis=1) 
        mean_action_entropies = jnp.sum(mean_action_distributions * jnp.log(mean_action_distributions), axis=-1)
        sorted_indices = jnp.argsort(action_entropies_reference)
        top_important_indices = sorted_indices[:int(self.action_entropies.shape[0] * 0.1)]
        important_states = jax.tree_util.tree_map(lambda x: x[top_important_indices], self.reshaped_states)
        top_100_imporant_indices = sorted_indices[:100]  
        batched_states = []
        top_1000_batched_states = []
        batched_chosen_actions = []
        top_1000_batched_chosen_actions = []
        for policy_id in range(self.action_entropies.shape[-1]):
            action_entropies = self.action_entropies[:, policy_id] 
            sorted_indices = jnp.argsort(action_entropies)
            top_important_indices = sorted_indices[:int(self.action_entropies.shape[0] * 0.01)]
            top_100_imporant_indices = sorted_indices[:100]
            top_1000_important_indices = sorted_indices[:1000]

            chosen_actions = self.chosen_actions[(top_100_imporant_indices, jnp.repeat(policy_id, top_100_imporant_indices.shape[0]))]
            top_1000_chosen_actions = self.chosen_actions[(top_1000_important_indices, jnp.repeat(policy_id, top_1000_important_indices.shape[0]))]
            batched_states.append(jax.tree_util.tree_map(lambda x: x[top_100_imporant_indices], self.reshaped_states))
            top_1000_batched_states.append(jax.tree_util.tree_map(lambda x: x[top_1000_important_indices], self.reshaped_states))
            batched_chosen_actions.append(chosen_actions)
            top_1000_batched_chosen_actions.append(top_1000_chosen_actions)
        sorted_indices = jnp.argsort(mean_action_entropies)
        top_100_important_indices = sorted_indices[:100]
        top_1000_important_indices = sorted_indices[:1000]
        majority_voted_states = jax.tree_util.tree_map(lambda x: x[top_1000_important_indices], self.reshaped_states)
        print("reshaped action distributions: ", self.reshaped_action_distributions.shape)
        all_actions = self.chosen_actions[top_1000_important_indices, :]
        bincounts = jnp.apply_along_axis(lambda x: jnp.bincount(x, length=self.reshaped_action_distributions.shape[-1]), axis=-1, arr=all_actions)
        majority_voted_chosen_actions = jnp.argmax(bincounts, axis=-1)
        return important_states, top_important_indices, batched_states, batched_chosen_actions, top_1000_batched_states, top_1000_batched_chosen_actions, majority_voted_states, majority_voted_chosen_actions 

    # TODO: Feature importance 

    def _plot_action_entropies(self, path_to_save): 
        
        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure()
        plt.scatter(self.action_entropies[:, -1], self.statewise_deviations, marker=".", alpha=0.01)
        plt.xlabel("action entropy")
        plt.ylabel("fraction of policy deviation from majority")
        plt.savefig(path_to_save + "action_entropies_vs_deviations.pdf", format="pdf", bbox_inches="tight")
        
    # def _get_most_important_states(self): 
    #     action_entropies = self.action_entropies[:, :, -1]
        
    #     sorted_action_entropies = action_entropies[sorted_indices]
    #     print(sorted_action_entropies[:10])
    #     return top_1000_important_indices 
        
        
        
        