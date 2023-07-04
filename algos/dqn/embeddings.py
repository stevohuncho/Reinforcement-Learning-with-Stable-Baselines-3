from stable_baselines3 import DQN
import torch as th

def get_dqn_embeddings(dqn_model: DQN, num_episodes: int):

    """
    Get embeddings obtained from the state-action pairs taken 
    in the trajectory of the passed model across num_episodes 
    episodes.
    """

    # Get the environment used by the model. We will be keeping 
    # track of the embeddings spanned by the model in its env.
    env = dqn_model.get_env()
    all_episode_embeddings = []
    all_episode_q_values = []

    # Get the model's policy
    policy = dqn_model.policy

    # Create an alias corresponding to the model's Q-network
    q_network = policy.q_net

    # Ensure that the model's policy is not training anymore
    policy.set_training_mode(False)

    # Repeat for the prescribed number of episodes
    for i in range(num_episodes):

        # Reset the environment
        episode_embeddings = []
        episode_q_values = []
        done = False
        obs = env.reset()

        # Keep going until episode finishes
        while not done:
            
            # Get the action that the model should take given the observation
            action_to_take, returned_state = dqn_model.predict(obs)

            # Convert the observation and action to tensor for embedding calculation
            tensor_obs, vectorized_env = policy.obs_to_tensor(obs)

            # Do not keep the gradient computation graph
            with th.no_grad():
      
                # Using the current observation and action to take, get the q-net 
                # embedding and score (of the first network if > 1 exist).
                q_values, q_embedding = q_network.forward(tensor_obs, last=True)

                # DQN predicts Q-value for each action. We only care about the action taken.
                q_value = th.squeeze(q_values)[action_to_take]

                # Squeeze the output to get rid of extra dimension
                q_value, q_embedding = th.squeeze(q_value), th.squeeze(q_embedding)
                episode_embeddings.append(q_embedding)
                episode_q_values.append(q_value)

            # Step the environment using the action. Here, action, rewards, 
            # and dones are arrays because we assume a vectorized env
            obs, reward, done, info = env.step(action_to_take)

        # Add this episode's embedding list here.
        all_episode_embeddings.append(th.vstack(episode_embeddings))
        all_episode_q_values.append(th.tensor(episode_q_values))

    full_embeddings = None
    full_q_vals = None
    
    for episode_embeddings, episode_q_vals in zip(all_episode_embeddings, all_episode_q_values):
        
        if full_embeddings is None:
            full_embeddings = episode_embeddings
        else:
            full_embeddings = th.cat([full_embeddings, episode_embeddings])
            
        if full_q_vals is None:
            full_q_vals = episode_q_vals
        else:
            full_q_vals = th.cat([full_q_vals, episode_q_vals])

    # Return the gathered embeddings
    return full_embeddings, full_q_vals