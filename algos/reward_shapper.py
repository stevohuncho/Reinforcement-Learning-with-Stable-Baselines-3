from torch import nn
import torch as th

class RewardShaper:

    """
    Base class for computing the auxiliary reward. We extend this class 
    to get the correct embeddings depending on the SB3 model.
    """

    def __init__(self, model, embeddings, associated_q_vals, gamma):

        """
        Creates a RewardShaper instance using the source model, generated 
        embeddings, their associated_q-vals, and the discount factor gamma
        """

        # Store embeddings, their associated q-vals, the learning rate gamma, 
        # and the SB3 model.
        self.embeddings = embeddings.to(model.device)
        self.associated_q_vals = associated_q_vals.to(model.device)
        self.gamma = gamma
        self.model = model

        # Normalize the embeddings. We want to compute cosine similarity, which 
        # is easily done by using the dot product if ||a|| = ||b|| = 1.
        self.embeddings = nn.functional.normalize(self.embeddings, p=2, dim=1)

    def _get_state_action_embedding(self, state, action):
        pass

    def _compute_phi_s_a(self, state, action):

        """
        Computes \Phi(s,a), which is used in the full reward shaping 
        F = \gamma \Phi(s',a') - \Phi(s,a). Here, we compute the 
        state-action embedding
        """

        # Get the embedding for this state-action. Normalize it.
        embedding = self._get_state_action_embedding(state, action)
        embedding = nn.functional.normalize(embedding, p=2, dim=0)

        # Get the dot product of each row of the embedding tensor with this 
        # normalized embedding. As each row vector is also normalized, the 
        # dot product gives exactly the cosine similarity. This can be done 
        # in bulk by doing a matrix product.
        similarity_scores = th.matmul(self.embeddings, embedding)

        # Lastly, we would like an average weighted q-val score. First, we 
        # compute the cosine-sim-weighted q-value sum using the dot product 
        # between our similarity scores and the associated q-vals for each 
        # embedding. Then, we average by the number of embeddings / q-vals
        sum_weighted_q_val_score = th.dot(self.associated_q_vals, similarity_scores)
        avg_weighted_q_val_score = sum_weighted_q_val_score / len(self.associated_q_vals)

        # Return the score.
        return avg_weighted_q_val_score.item()

    def get_auxiliary_reward(self, state, action, next_state, next_action):

        # Get \Phi(s,a) and \Phi(s',a')
        phi_s_a = self._compute_phi_s_a(state, action)
        next_phi_s_a = self._compute_phi_s_a(next_state, next_action)
        
        # Compute auxiliary reward F = \gamma \Phi(s',a') - \Phi(s,a)
        aux_reward = self.gamma * next_phi_s_a - phi_s_a
        return aux_reward
