import pandas as pd 
import os 
import pickle 
import jax 
import jax.numpy as jnp 
import numpy as np 


def load_parameters(policy_path):
    parameters = jnp.load(
        policy_path,
        allow_pickle=True,
    )
    return parameters

def load_stacked_parameters(policy_indices, path_to_policies):
    policy_paths = [
        path_to_policies + f"policy-{p}.pkl"
        for p in policy_indices
    ]
    params_list = [load_parameters(p) for p in policy_paths]
    params_stacked = jax.tree.map(lambda *args: jnp.stack(args), *params_list)
    return params_stacked, params_list 


def load_rashomon_set(policy_path, evaluation_data_path, alpha, max_num_policies): 
    df = pd.read_csv(evaluation_data_path)
    rewards = df["mean_rewards"].values
    sorted_indices_by_rewards = jnp.argsort(rewards)[-(max_num_policies):] 
    sorted_rewards = rewards[sorted_indices_by_rewards] 
    selected_indices = df["policy_index"].values[sorted_indices_by_rewards]
    largest_rewards = max(sorted_rewards)
    threshold = largest_rewards * (1 - alpha)
    policy_indices = selected_indices[sorted_rewards >= threshold]
    selected_rewards = sorted_rewards[sorted_rewards >= threshold]
    #print(selected_rewards)
    print("Best set of policies from the freely trained policies: ")
    print(f"Max reward: {largest_rewards:.2f}")
    print(f"Min reward: {min(selected_rewards):.2f}")
    print(f"Mean reward: {jnp.mean(selected_rewards):.2f}")
    print(f"Std reward: {jnp.std(selected_rewards):.2f}")
    print(f"Alpha threshold: {threshold:.2f}")
    print(f"Number of policies: {len(policy_indices)}")
    
    params_stacked, params_list = load_stacked_parameters(policy_indices, policy_path)
    return params_stacked, params_list

def load_best_policies(policy_path, evaluation_data_path, max_num_policies): 
    df = pd.read_csv(evaluation_data_path)
    rewards = df["mean_rewards"].values
    sorted_indices_by_rewards = jnp.argsort(rewards)[-(max_num_policies):] 
    sorted_rewards = rewards[sorted_indices_by_rewards] 
    selected_indices = df["policy_index"].values[sorted_indices_by_rewards]
    largest_rewards = max(sorted_rewards)
    policy_indices = selected_indices
    selected_rewards = sorted_rewards
    #print(selected_rewards)
    # print("Best set of policies from the freely trained policies: ")
    # print(f"Max reward: {largest_rewards:.2f}")
    # print(f"Min reward: {min(selected_rewards):.2f}")
    # print(f"Mean reward: {jnp.mean(selected_rewards):.2f}")
    # print(f"Std reward: {jnp.std(selected_rewards):.2f}")
    # print(f"Number of policies: {len(policy_indices)}")
    
    params_stacked, params_list = load_stacked_parameters(policy_indices, policy_path)
    return params_stacked, params_list, sorted_rewards


def load_similar_policies(policy_path, evaluation_data_path, mean_rewards, training_setup): 
    df = pd.read_csv(evaluation_data_path)
    rewards = jnp.array(df["mean_rewards"].values)
    policy_indices = []
    chosen_mean_rewards = []
    for mean_reward in mean_rewards:
        count = 0 
        differences = rewards - mean_reward
        chosen_index = jnp.argmin(jnp.abs(differences))
        while chosen_index in policy_indices:
            chosen_index = jnp.argsort(jnp.abs(differences))[count + 1]
            count += 1
        policy_indices.append(chosen_index)
        chosen_mean_rewards.append(rewards[chosen_index])
    
    # print(f"Similar policies set from the setup {training_setup}: ")
    # print(f"Max reward: {max(chosen_mean_rewards):.2f}")
    # print(f"Min reward: {min(chosen_mean_rewards):.2f}")
    # print(f"Mean reward: {np.mean(chosen_mean_rewards):.2f}")
    # print(f"Std reward: {np.std(chosen_mean_rewards):.2f}")
    # print(f"Number of policies: {len(policy_indices)}")
    # print(f"Policy indices: {policy_indices}")
    
    params_stacked, params_list = load_stacked_parameters(policy_indices, policy_path)
    return params_stacked, params_list, chosen_mean_rewards