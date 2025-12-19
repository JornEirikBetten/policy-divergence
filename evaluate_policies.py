import jax 
import jax.numpy as jnp
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import os
import sys
import time
import random
import distrax
import pgx
import pickle

from src.model.model import get_model


def build_evaluate_fn(env, forward_pass, num_eval_envs): 
    def evaluate(params, rng_key): 
        step_fn = jax.vmap(env.step)
        rng_key, sub_key = jax.random.split(rng_key)
        subkeys = jax.random.split(sub_key, num_eval_envs)
        state = jax.vmap(env.init)(subkeys)
        R = jnp.zeros_like(state.rewards)

        def cond_fn(tup):
            state, _, _ = tup
            return ~state.terminated.all()
        
        def loop_fn(tup):
            state, R, rng_key = tup
            logits, value = forward_pass.apply(params, state.observation)
            pi = distrax.Categorical(logits=logits)
            rng_key, _rng = jax.random.split(rng_key)
            action = pi.sample(seed=_rng)
            rng_key, _rng = jax.random.split(rng_key)
            keys = jax.random.split(_rng, state.observation.shape[0])
            state = step_fn(state, action, keys)
            return state, R + state.rewards, rng_key
        state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
        return R.mean(), R.std() 
    return evaluate


#env_names = ["minatar-asterix", "minatar-breakout", "minatar-freeway", "minatar-seaquest", "minatar-space_invaders"]
rng_key = jax.random.PRNGKey(0)
num_eval_envs = 1024


dw_policies_path = os.getcwd() + "/different-world-policies/"
sw_policies_path = os.getcwd() + "/same-world-policies/"
dwsi_policies_path = os.getcwd() + "/different-world-same-init-policies/"


# results = {
#     "minatar-asterix-mean": np.ones(500) * -1.0,
#     "minatar-breakout-mean": np.ones(500) * -1.0,
#     "minatar-freeway-mean": np.ones(500) * -1.0,
#     "minatar-seaquest-mean": np.ones(500) * -1.0,
#     "minatar-space_invaders-mean": np.ones(500) * -1.0,
#     "minatar-asterix-std": np.ones(500) * -1.0,
#     "minatar-breakout-std": np.ones(500) * -1.0,
#     "minatar-freeway-std": np.ones(500) * -1.0,
#     "minatar-seaquest-std": np.ones(500) * -1.0,
#     "minatar-space_invaders-std": np.ones(500) * -1.0,
#     "policy_index": np.arange(1, 501)
# }
results = {
    "mean_rewards": np.ones(500) * -1.0,
    "std_rewards": np.ones(500) * -1.0,
    "policy_index": np.arange(1, 501)
}


env_name = sys.argv[1]
setup = sys.argv[2]
if setup == "different-world":
    policies_path = dw_policies_path
    save_path = os.getcwd() + "/evaluation_of_policies/" + env_name  + "/"
    csv_name = "different-world-policy-performances.csv"
elif setup == "same-world":
    policies_path = sw_policies_path
    save_path = os.getcwd() + "/evaluation_of_policies/" + env_name + "/"
    csv_name = "same-world-policy-performances.csv"
elif setup == "different-world-same-init":
    policies_path = dwsi_policies_path
    save_path = os.getcwd() + "/evaluation_of_policies/" + env_name + "/"
    csv_name = "different-world-same-init-policy-performances.csv"
else:
    raise ValueError(f"Invalid setup: {setup}")
#for env_name in env_names:
env = pgx.make(env_name)
forward_pass = get_model(env.num_actions)
forward_pass = hk.without_apply_rng(hk.transform(forward_pass))

evaluate_fn = build_evaluate_fn(env, forward_pass, num_eval_envs)
path_to_environment_policies = policies_path + env_name + "/"
for i in range(1, 201): 
    policy_path = path_to_environment_policies + f"policy-{i}.pkl"
    if os.path.exists(policy_path): 
        with open(policy_path, "rb") as f:
            params = pickle.load(f)
        #rng_key, sub_key = jax.random.split(rng_key)
        rewards_mean, rewards_std = evaluate_fn(params, rng_key)
        print(f"Policy {i} has mean reward {rewards_mean} and std {rewards_std}")
        results["mean_rewards"][i-1] = rewards_mean
        results["std_rewards"][i-1] = rewards_std
    else: 
        rewards_mean = -1.0 
        print(f"Policy {i} does not exist")
            

df = pd.DataFrame(results)
#save_path = os.getcwd() + "/evaluation_of_policies/" + env_name + "/" 
if not os.path.exists(save_path): 
    os.makedirs(save_path)  
df.to_csv(save_path + csv_name, index=False)