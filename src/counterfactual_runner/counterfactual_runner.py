import jax 
import pgx 
import jax.numpy as jnp 

# TODO: Counterfactual runner 

def _reshape_states(x): 
    if len(x.shape) == 3: 
        if x.shape[2] == 0: 
            return x.reshape((x.shape[0]*x.shape[1], 0))
        else: 
            return x.reshape((-1,) + x.shape[2:])
    else: 
        return x.reshape((-1,) + x.shape[2:])


def build_counterfactual_runner(forward_pass, env_fn, max_steps, num_repetitions): 
    
    #num_actions = env_fn.num_actions 
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def counterfactual_runner(parameters, state, first_action): 
        rng = jax.random.PRNGKey(1234)
        num_states = state.observation.shape[0]
        actions = jnp.repeat(first_action, num_repetitions)
        rng = jax.random.PRNGKey(1234)
        rng, _rng = jax.random.split(rng) 
        rngs = jax.random.split(_rng, num_repetitions) 
        states = jax.tree.map(lambda x: jnp.stack([x] * num_repetitions), state)
        states = jax.tree.map(lambda x: jnp.concatenate(x, axis=0), states)
        #print(states.observation.shape)
        #states = jax.tree.map(_reshape_states, states)
        # Initial action
        next_states = jax.vmap(step_fn, in_axes=(0, None, None))(states, actions, rngs)
        
        
        
        
        def cond_fn(tup): 
            states, step, R, rng = tup 
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup): 
            states, step, R, rng = tup
            majority_action, action, action_distribution = jax.vmap(forward_pass, in_axes=(None, 0))(parameters, states)
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, num_repetitions)
            # print("Rngs: ", rngs.shape)
            # print("Majority action: ", majority_action.shape)
            # print("Observation: ", states.observation.shape)
            # print("Action distribution: ", action_distribution.shape)
            next_states = jax.vmap(step_fn, in_axes=(0, 0, None))(states, majority_action, rngs)
            R += jnp.where(next_states.terminated, 0, next_states.rewards) 
            step += 1 
            return next_states, step, R, rng 
        
        
        R = jnp.zeros_like(next_states.rewards)
        step = 0 
        rng = jax.random.PRNGKey(1234)
        states = next_states 
        states, step, R, rng = jax.lax.while_loop(cond_fn, body_fn, (states, step, R, rng))
        return R 
    return counterfactual_runner 
        
        

def build_alternative_counterfactual_runner(forward_pass, env_fn, max_steps, num_repetitions, num_policies): 
    
    #num_actions = env_fn.num_actions 
    step_fn = jax.jit(jax.vmap(env_fn.step))
    def counterfactual_runner(parameters, state, first_action): 
        # rng = jax.random.PRNGKey(1234)
        # num_states = state.observation.shape[0]
        # actions = jnp.repeat(first_action, num_repetitions)
        # rng = jax.random.PRNGKey(1234)
        # rng, _rng = jax.random.split(rng) 
        # rngs = jax.random.split(_rng, num_repetitions) 
        # states = jax.tree.map(lambda x: jnp.stack([x] * num_repetitions), state)
        # states = jax.tree.map(lambda x: x.reshape((x.shape[1], x.shape[0], ) + x.shape[2:]), states)
        # #print(states.observation.shape)
        # #states = jax.tree.map(_reshape_states, states)
        # # Initial action
        # next_states = jax.vmap(step_fn, in_axes=(0, None, None))(states, actions, rngs)
        
        
        
        
        def cond_fn(tup): 
            states, step, R, rng = tup 
            return jnp.logical_and(~(state.terminated).all(), step <= max_steps)
        
        def body_fn(tup): 
            states, step, R, rng = tup
            #majority_action, action, action_distribution = jax.vmap(forward_pass, in_axes=(None, 0))(parameters, states)
            
            majority_action, action, action_distribution = jax.lax.cond(
                step == 0, 
                lambda: 
                    (first_action*jnp.ones((states.observation.shape[:2]), dtype=jnp.int32), jnp.zeros((states.observation.shape[0], num_policies, num_repetitions), dtype=jnp.int32), jnp.zeros((states.observation.shape[:2] + (num_policies,env_fn.num_actions)))),
                lambda: 
                    jax.vmap(forward_pass, in_axes=(None, 0))(parameters, states),
            )
            rng, _rng = jax.random.split(rng)
            rngs = jax.random.split(_rng, num_repetitions)
            # print("Majority action: ", majority_action.shape)
            # print("Action: ", action.shape)
            # print("rngs: ", rngs.shape)
            
            next_states = jax.vmap(step_fn, in_axes=(0, 0, None))(states, majority_action, rngs)
            R += jnp.where(next_states.terminated, 0, next_states.rewards.squeeze(axis=-1)) 
            step += 1 
            return next_states, step, R, rng 
            
            # rng, _rng = jax.random.split(rng)
            # rngs = jax.random.split(_rng, num_repetitions)
            # # print("Rngs: ", rngs.shape)
            # # print("Majority action: ", majority_action.shape)
            # # print("Observation: ", states.observation.shape)
            # # print("Action distribution: ", action_distribution.shape)
            # next_states = jax.vmap(step_fn, in_axes=(0, 0, None))(states, majority_action, rngs)
            # R += next_states.rewards 
            # step += 1 
            # return next_states, step, R, rng 
        
        states = jax.tree.map(lambda x: jnp.stack([x] * num_repetitions), state)
        print("States: ", states.observation.shape)
        states = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), states)
        print("States: ", states.observation.shape)
        #next_states = jax.vmap(step_fn, in_axes=(0, None, None))(states, first_action, rngs)
        R = jnp.zeros_like(states.rewards.squeeze(axis=-1))
        step = 0 
        rng = jax.random.PRNGKey(1234)
        states, step, R, rng = jax.lax.while_loop(cond_fn, body_fn, (states, step, R, rng))
        return R 
    return counterfactual_runner