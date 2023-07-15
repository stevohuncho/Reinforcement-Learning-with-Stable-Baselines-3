from typing import Any, Dict, Optional, Tuple, Type, Union
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.dqn.policies import DQNPolicy
from transfer_learning.map_resizer import MapResizer
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

class FrozenLakeTransferDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        predict_model: DQN,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy, 
            env, 
            learning_rate, 
            buffer_size, 
            learning_starts, 
            batch_size, 
            tau, 
            gamma, 
            train_freq, 
            gradient_steps, 
            replay_buffer_class, 
            replay_buffer_kwargs, 
            optimize_memory_usage, 
            target_update_interval, 
            exploration_fraction, 
            exploration_initial_eps, 
            exploration_final_eps, 
            max_grad_norm, 
            stats_window_size, 
            tensorboard_log, 
            policy_kwargs, 
            verbose, 
            seed, 
            device, 
            _init_setup_model
        )
        self.predict_model = predict_model
    
    def learn():
        super().learn()
        pass