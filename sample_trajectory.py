import jax 
import jax.numpy as jnp 
import sys 
import os 
import pgx 
import haiku as hk 
from src.policy_loader.policy_loader import load_rashomon_set, load_best_policies, load_similar_policies
from src.model.model import get_model
from src.simulator.simulator import Trajectory, TwoSetTrajectory, ThreeSetTrajectory
from src.simulator.simulator import build_standard_pgx_simulator, build_two_set_pgx_simulator, build_three_set_pgx_simulator
import pickle 


def load_sets_of_parameters(env_name): 
    path_to_same_world_policies = os.getcwd() + "/same-world-policies/" + env_name + "/"
    path_to_different_world_policies = os.getcwd() + "/different-world-policies/" + env_name + "/"
    path_to_same_world_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/same-world-policy-performances.csv"
    path_to_different_world_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-policy-performances.csv"
    path_to_different_world_same_init_policies = os.getcwd() + "/different-world-same-init-policies/" + env_name + "/"
    path_to_different_world_same_init_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-same-init-policy-performances.csv"
    different_world_params_stacked, different_world_params_list, different_world_rewards = load_best_policies(path_to_different_world_policies, path_to_different_world_evaluation_data, 20)
    same_world_params_stacked, same_world_params_list, same_world_rewards = load_similar_policies(path_to_same_world_policies, path_to_same_world_evaluation_data, different_world_rewards, "SWDI")
    different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards = load_similar_policies(path_to_different_world_same_init_policies, path_to_different_world_same_init_evaluation_data, different_world_rewards, "DWSI")
    return same_world_params_stacked, same_world_params_list, different_world_params_stacked, different_world_params_list, different_world_same_init_params_stacked, different_world_same_init_params_list


env_name = sys.argv[1]
same_world_params_stacked, same_world_params_list, different_world_params_stacked, different_world_params_list, different_world_same_init_params_stacked, different_world_same_init_params_list = load_sets_of_parameters(env_name)
env_fn = pgx.make(env_name)
forward_pass = get_model(env_fn.num_actions)
forward_pass = hk.without_apply_rng(hk.transform(forward_pass))

def majority_vote_forward_pass(different_world_params_stacked, same_world_params_stacked, different_world_same_init_params_stacked, state): 
    logits, value = jax.vmap(forward_pass.apply, in_axes=(0, None))(different_world_params_stacked, state.observation)
    logits_same_world, value_same_world = jax.vmap(forward_pass.apply, in_axes=(0, None))(same_world_params_stacked, state.observation)
    logits_different_world_same_init, value_different_world_same_init = jax.vmap(forward_pass.apply, in_axes=(0, None))(different_world_same_init_params_stacked, state.observation)
    first_action_distributions = jax.nn.softmax(logits, axis=-1)
    #print("Action distributions: ", first_action_distributions.shape)
    second_action_distributions = jax.nn.softmax(logits_same_world, axis=-1)
    #print("Second action distributions: ", second_action_distributions.shape)
    third_action_distributions = jax.nn.softmax(logits_different_world_same_init, axis=-1)
    actions = logits.argmax(axis=-1)
    bincounts = jnp.apply_along_axis(lambda x: jnp.bincount(x, length=env_fn.num_actions), axis=0, arr=actions)
    majority_action = jnp.argmax(bincounts, axis=0)
    return majority_action, actions, jnp.swapaxes(first_action_distributions, 0, 1), jnp.swapaxes(second_action_distributions, 0, 1), jnp.swapaxes(third_action_distributions, 0, 1)

simulator = build_three_set_pgx_simulator(env_fn, majority_vote_forward_pass, 2500, 20)

num_initial_states = 100
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, num_initial_states)
states = jax.vmap(env_fn.init)(subkeys)

trajectories = simulator(different_world_params_stacked, same_world_params_stacked, different_world_same_init_params_stacked, states, key)
print("Trajectories: ", trajectories.state.observation.shape)
print("Actions: ", trajectories.action.shape)
print("Accumulated rewards: ", trajectories.accumulated_rewards.shape)
print("DWDI action distribution: ", trajectories.dwdi_action_distribution.shape)
print("SWDI action distribution: ", trajectories.swdi_action_distribution.shape)
print("DWSI action distribution: ", trajectories.dwsi_action_distribution.shape)
print("DWDI: ", trajectories.dwdi_action_distribution[10, 0, 0, :])
print("SWDI: ", trajectories.swdi_action_distribution[10, 0, 0, :])
print("DWSI: ", trajectories.dwsi_action_distribution[10, 0, 0, :])
print("Randomness: ", trajectories.randomness.shape)


trajectory_path = os.getcwd() + "/trajectories/" + env_name + "/" 
if not os.path.exists(trajectory_path):
    os.makedirs(trajectory_path)
with open(trajectory_path + "same-world-trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)
# with open(os.getcwd() + "/trajectories/" + env_name + "/same-world-trajectories.pkl", "wb") as f:
#     pickle.dump(trajectories, f)