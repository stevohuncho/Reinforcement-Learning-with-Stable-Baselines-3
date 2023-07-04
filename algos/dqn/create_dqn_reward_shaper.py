from stable_baselines3 import DQN
from .embeddings import get_dqn_embeddings
from algos.reward_shapper import RewardShaper
import torch as th
from stable_baselines3.common.vec_env import VecEnv

def create_dqn_reward_shaper(source_model: DQN, num_sampling_episodes: int):
    
    embeddings, q_vals = get_dqn_embeddings(source_model, num_sampling_episodes)
    reward_shaper = DQNRewardShaper(source_model, embeddings, q_vals, source_model.gamma)
    return reward_shaper

class DQNRewardShaper(RewardShaper):

    def __init__(self, model, embeddings, associated_q_vals, gamma):

        super(DQNRewardShaper, self).__init__(model, embeddings, associated_q_vals, gamma)

    def _get_state_action_embedding(self, state, action):

        # Get the model's policy
        policy = self.model.policy

        # Create an alias corresponding to the model's Q-network
        q_network = policy.q_net

        # Ensure that the model's policy is not training anymore
        policy.set_training_mode(False)

        # Convert the observation and action to tensor for embedding calculation
        if type(state) != th.Tensor:
            tensor_obs, vectorized_env = policy.obs_to_tensor(state)
        else:
            tensor_obs = state
        tensor_obs = th.reshape(tensor_obs, (1,-1)).to(self.model.device)

        # Do not keep the gradient computation graph
        with th.no_grad():
      
            # Using the current observation and action to take, get the q-net 
            # embedding and score (of the first network if > 1 exist).
            _, q_embedding = q_network.forward(tensor_obs, last=True)

            # Squeeze the output to get rid of extra dimension
            q_embedding = th.squeeze(q_embedding)

            # Return the embedding
            return q_embedding