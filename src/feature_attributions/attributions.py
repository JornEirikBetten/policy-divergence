import jax 
import jax.numpy as jnp 
import haiku as hk 
import chex 
import pgx 
from typing import NamedTuple, Callable, Tuple
from dataclasses import fields 

class Attributions(NamedTuple): 
    state: pgx.State
    action: chex.Array
    accumulated_rewards: chex.Array
    action_distribution: chex.Array