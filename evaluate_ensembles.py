import jax 
import jax.numpy as jnp  
import sys 
import os 
import pgx 
import haiku as hk 
from src.policy_loader.policy_loader import load_rashomon_set, load_best_policies, load_similar_policies
from src.model.model import get_model


def evaluate_ensemble(env_name, params_stacked): 
    print("Evaluating ensemble of policies for environment: ", env_name)
    max_steps = 2500 
    env_fn = pgx.make(env_name)
    forward_pass = get_model(env_fn.num_actions)
    forward_pass = hk.without_apply_rng(hk.transform(forward_pass))

    def majority_vote_action(stacked_params, state): 
        logits, value = jax.vmap(forward_pass.apply, in_axes=(0, None))(stacked_params, state.observation)
        action_distributions = jax.nn.softmax(logits, axis=-1)
        actions = logits.argmax(axis=-1)
        bincounts = jnp.apply_along_axis(lambda x: jnp.bincount(x, length=env_fn.num_actions), axis=0, arr=actions)
        majority_action = jnp.argmax(bincounts, axis=0)
        return majority_action, actions, action_distributions 
    
    def build_evaluate_fn(env_fn, decision_fn, num_eval_envs): 
        step_fn = jax.jit(jax.vmap(env_fn.step)) 
        def evaluate(params, rng_key): 
            def cond_fn(tup): 
                state, R, rng_key = tup 
                return ~state.terminated.all()
            
            def loop_fn(tup): 
                state, R, rng_key = tup 
                majority_action, actions, action_distributions = decision_fn(params, state)
                rng_key, _rng = jax.random.split(rng_key)
                keys = jax.random.split(_rng, state.observation.shape[0])
                state = step_fn(state, majority_action, keys)
                return state, R + state.rewards.squeeze(axis=-1), rng_key 

            rng, _rng = jax.random.split(rng_key)
            rngs = jax.random.split(_rng, num_eval_envs)
            states = jax.vmap(env_fn.init)(rngs)
            R = jnp.zeros_like(states.rewards.squeeze(axis=-1))
            states, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (states, R, rng))
            return R.mean(), R.std() 
        return jax.jit(evaluate) 

    num_eval_envs = 1024
    evaluate_fn = build_evaluate_fn(env_fn, majority_vote_action, num_eval_envs) 

    rng = jax.random.PRNGKey(123123)

    mean, std = evaluate_fn(params_stacked, rng)
    return mean, std 

def load_policies(env_name):
    path_to_policies = os.getcwd() + "/different-world-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-policy-performances.csv"
    different_world_params_stacked, different_world_params_list, different_world_rewards = load_best_policies(path_to_policies, path_to_evaluation_data, 20)
    path_to_policies = os.getcwd() + "/same-world-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/same-world-policy-performances.csv"
    same_world_params_stacked, same_world_params_list, same_world_rewards = load_similar_policies(path_to_policies, path_to_evaluation_data, different_world_rewards, "swdi")
    path_to_policies = os.getcwd() + "/different-world-same-init-policies/" + env_name + "/"
    path_to_evaluation_data = os.getcwd() + "/evaluation_of_policies/" + env_name + "/different-world-same-init-policy-performances.csv"
    different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards = load_similar_policies(path_to_policies, path_to_evaluation_data, different_world_rewards, "dwsi")
    return different_world_params_stacked, different_world_params_list, same_world_params_stacked, same_world_params_list, different_world_rewards, same_world_rewards, different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards

env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"] 

results = {
    "env_name": [],
    "policy_type": [],
    "mean_rewards": [],
    "std_rewards": []
}
all_different_world_stacked_params = []
all_same_world_stacked_params = []
all_different_world_same_init_stacked_params = []
for env_name in env_names: 
    different_world_params_stacked, different_world_params_list, same_world_params_stacked, same_world_params_list, different_world_rewards, same_world_rewards, different_world_same_init_params_stacked, different_world_same_init_params_list, different_world_same_init_rewards = load_policies(env_name)
    all_different_world_stacked_params.append(different_world_params_stacked)
    all_same_world_stacked_params.append(same_world_params_stacked)
    all_different_world_same_init_stacked_params.append(different_world_same_init_params_stacked)


import pandas as pd 

# df = pd.DataFrame(results)
save_path = os.getcwd() + "/evaluation_of_policies/mv_ensemble_results.csv"
# df.to_csv(save_path, index=False)

data = pd.read_csv(save_path)
means = data["mean_rewards"].values
stds = data["std_rewards"].values 
performance_datas = {
    "different-world": "different-world-policy-performances.csv",
    "same-world": "same-world-policy-performances.csv",
    "different-world-same-init": "different-world-same-init-policy-performances.csv",
}
k = 0 
for i, env_name in enumerate(env_names): 
    different_world_params = all_different_world_stacked_params[i]
    #jax.tree.map(lambda x: print(x.shape), different_world_params)
    mean_free, std_free = evaluate_ensemble(env_name, different_world_params)
    results["env_name"].append(env_name)
    results["policy_type"].append("different-world")
    results["mean_rewards"].append(mean_free)
    results["std_rewards"].append(std_free)
    same_world_params = all_same_world_stacked_params[i]
    mean_same_world, std_same_world = evaluate_ensemble(env_name, same_world_params)
    results["env_name"].append(env_name)
    results["policy_type"].append("same-world")
    results["mean_rewards"].append(mean_same_world)
    results["std_rewards"].append(std_same_world)
    different_world_same_init_params = all_different_world_same_init_stacked_params[i]
    mean_different_world_same_init, std_different_world_same_init = evaluate_ensemble(env_name, different_world_same_init_params)
    results["env_name"].append(env_name)
    results["policy_type"].append("different-world-same-init")
    results["mean_rewards"].append(mean_different_world_same_init)
    results["std_rewards"].append(std_different_world_same_init)
    print(f"{env_name}: DWDI mean: {mean_free}, DWDI SEM: {1.96*std_free/jnp.sqrt(1024)}, DWSI mean: {mean_different_world_same_init}, DWSI SEM: {1.96*std_different_world_same_init/jnp.sqrt(1024)}, SWDI mean: {mean_same_world}, DWSI SEM: {1.96*std_same_world/jnp.sqrt(1024)}")
df = pd.DataFrame(results)
df.to_csv(save_path, index=False)










