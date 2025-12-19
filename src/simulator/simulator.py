import jax 
import pgx 
import jax.numpy as jnp 
import pandas as pd 
import chex 
import haiku as hk 
from typing import NamedTuple, Callable, Tuple

class Trajectory(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    action_distribution: chex.Array
    randomness: jax.random.PRNGKey
    
class TwoSetTrajectory(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    first_action_distribution: chex.Array
    second_action_distribution: chex.Array
    randomness: jax.random.PRNGKey
    
class ThreeSetTrajectory(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    dwdi_action_distribution: chex.Array
    swdi_action_distribution: chex.Array
    dwsi_action_distribution: chex.Array
    randomness: jax.random.PRNGKey

def build_standard_pgx_simulator(env_fn: Callable, forward_pass: Callable, max_steps: int, num_policies: int) -> Callable: 
    """
    Builds a standard simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def simulator(params: hk.Params, state: pgx.State, key: jax.random.PRNGKey) -> Trajectory: 
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> bool: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]) -> Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]: 
            state, key, step, states, actions, action_distributions, rewards, rngs = tup
            majority_action, action, action_distribution = forward_pass(params, state)
            # Sample next state 
            key, _key = jax.random.split(key)
            keys = jax.random.split(_key, state.observation.shape[0])
            rngs = rngs.at[step].set(keys)
            new_state = step_fn(state, majority_action, keys)
            rewards = rewards + state.rewards.squeeze()
            states = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), states, state # Get old state where action is taken
            )
            actions = actions.at[step].set(majority_action)
            action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), action_distributions, action_distribution
            )
            return (new_state, key, step + 1, states, actions, action_distributions, rewards, rngs) 
        
        # Setup data to extract
        num_states = state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), state
        )
        action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        rewards = jnp.zeros(num_states)
        rngs = jnp.zeros((max_steps, num_states, 2))
        step = 0
        # Run loop 
        state, key, step, states, actions, action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (state, key, step, states, actions, action_distributions, rewards, rngs))
        
        trajectory = Trajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            action_distribution=action_distributions, 
            randomness=rngs)
        
        return trajectory
    return simulator


def build_two_set_pgx_simulator(env_fn: Callable, forward_pass: Callable, max_steps: int, num_policies: int) -> Callable: 
    """
    Builds a standard simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def simulator(params: hk.Params, second_params: hk.Params, state: pgx.State, key: jax.random.PRNGKey) -> Trajectory: 
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> bool: 
            state, key, step, states, actions, first_action_distributions, second_action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]) -> Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]: 
            state, key, step, states, actions, first_action_distributions, second_action_distributions, rewards, rngs = tup
            majority_action, action, first_action_distribution, second_action_distribution = forward_pass(params, second_params, state)
            # Sample next state 
            key, _key = jax.random.split(key)
            keys = jax.random.split(_key, state.observation.shape[0])
            rngs = rngs.at[step].set(keys)
            new_state = step_fn(state, majority_action, keys)
            rewards = rewards + state.rewards.squeeze()
            states = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), states, state # Get old state where action is taken
            )
            actions = actions.at[step].set(majority_action)
            first_action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), first_action_distributions, first_action_distribution
            )
            second_action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), second_action_distributions, second_action_distribution
            )
            return (new_state, key, step + 1, states, actions, first_action_distributions, second_action_distributions, rewards, rngs) 
        
        # Setup data to extract
        num_states = state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), state
        )
        first_action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        second_action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        rewards = jnp.zeros(num_states)
        rngs = jnp.zeros((max_steps, num_states, 2))
        step = 0
        # Run loop 
        state, key, step, states, actions, first_action_distributions, second_action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (state, key, step, states, actions, first_action_distributions, second_action_distributions, rewards, rngs))
        
        trajectory = TwoSetTrajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            first_action_distribution=first_action_distributions, 
            second_action_distribution=second_action_distributions, 
            randomness=rngs)
        
        return trajectory
    return simulator

def build_three_set_pgx_simulator(env_fn: Callable, forward_pass: Callable, max_steps: int, num_policies: int) -> Callable: 
    """
    Builds a standard simulator that runs a policy for maximally max_steps interaction steps with the environment and returns the trajectory. 
    """
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def simulator(params: hk.Params, second_params: hk.Params, third_params: hk.Params, state: pgx.State, key: jax.random.PRNGKey) -> Trajectory: 
        def cond_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int]) -> bool: 
            state, key, step, states, actions, dwdi_action_distributions, swdi_action_distributions, dwsi_action_distributions, rewards, rngs = tup
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup: Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]) -> Tuple[pgx.State, jax.random.PRNGKey, int, chex.Array, chex.Array, chex.Array, chex.Array]: 
            state, key, step, states, actions, dwdi_action_distributions, swdi_action_distributions, dwsi_action_distributions, rewards, rngs = tup
            majority_action, action, dwdi_action_distribution, swdi_action_distribution, dwsi_action_distribution = forward_pass(params, second_params, third_params, state)
            # Sample next state 
            key, _key = jax.random.split(key)
            keys = jax.random.split(_key, state.observation.shape[0])
            rngs = rngs.at[step].set(keys)
            new_state = step_fn(state, majority_action, keys)
            rewards = rewards + state.rewards.squeeze()
            states = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), states, state # Get old state where action is taken
            )
            actions = actions.at[step].set(majority_action)
            dwdi_action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), dwdi_action_distributions, dwdi_action_distribution
            )
            swdi_action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), swdi_action_distributions, swdi_action_distribution
            )
            dwsi_action_distributions = jax.tree.map(
                lambda arr, x: arr.at[step].set(x), dwsi_action_distributions, dwsi_action_distribution
            )
            return (new_state, key, step + 1, states, actions, dwdi_action_distributions, swdi_action_distributions, dwsi_action_distributions, rewards, rngs) 
        
        # Setup data to extract
        num_states = state.observation.shape[0]; num_actions = env_fn.num_actions
        actions = -jnp.ones(
            (max_steps, num_states)
        )
        states = jax.tree.map(
            lambda x: jnp.ones((max_steps,) + x.shape, x.dtype), state
        )
        dwdi_action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        swdi_action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        dwsi_action_distributions = jnp.zeros(
            (max_steps, num_states, num_policies, num_actions)
        )
        rewards = jnp.zeros(num_states)
        rngs = jnp.zeros((max_steps, num_states, 2))
        step = 0
        # Run loop 
        state, key, step, states, actions, dwdi_action_distributions, swdi_action_distributions, dwsi_action_distributions, rewards, rngs = jax.lax.while_loop(
            cond_fn, 
            body_fn, 
            (state, key, step, states, actions, dwdi_action_distributions, swdi_action_distributions, dwsi_action_distributions, rewards, rngs))
        
        trajectory = ThreeSetTrajectory(
            state=states, 
            action=actions, 
            accumulated_rewards=rewards, 
            dwdi_action_distribution=dwdi_action_distributions, 
            swdi_action_distribution=swdi_action_distributions, 
            dwsi_action_distribution=dwsi_action_distributions, 
            randomness=rngs)
        
        return trajectory
    return simulator