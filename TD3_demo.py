import traceback
import ray
import gym
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import pdb
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ddpg import ddpg, td3
from ray.rllib.agents.ddpg.td3 import TD3_DEFAULT_CONFIG

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.observation_space = gym.spaces.Box(low=-np.float('inf'), high=np.float('inf'), shape=(10, ), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float64)
    def reset(self):
        return np.random.randn(10)
    def step(self, action):
        obs = np.random.randn(10)
        reward = action + np.random.randn(1)
        done = False
        return obs, float(reward), False, {'info': 'None'}

class MyCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict = None):
        super().__init__(legacy_callbacks_dict)
        self.accumulated_call_on_episode_step = 0
        self.accumulated_call_on_episode_end = 0
        print("init MyCallbacks")

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs) -> None:
        # print("callback on episode step")
        self.accumulated_call_on_episode_step += 1
        pass
    
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs) -> None:
        self.accumulated_call_on_episode_end += 1
        print(f"{self.accumulated_call_on_episode_end} callbacks on episode end, {self.accumulated_call_on_episode_step} callbacks on episode step")
        pass

def main():
    try:
        # 1. 将ADMMSolverEnv注册到RLlib框架，方便后续通过env名字配置训练需要交互的env
        register_env("MyEnv", MyEnv)

        # 2. 配置TD3算法的训练参数
        TD3config = TD3_DEFAULT_CONFIG.copy()
        # Basic configs
        TD3config['framework'] = "torch"
        TD3config['num_gpus'] = 0  # 使用的GPU数量
        TD3config['num_cpus_per_worker'] = 3  # 每个RolloutWorker分配的CPU核数
        TD3config['num_workers'] = 1  # 使用多少个RolloutWorkers并发与env交互，获取训练样本
        TD3config['env'] = "MyEnv"
        TD3config['log_level'] = "INFO"  # 训练过程的日志级别
        TD3config["logger_config"] = {"type": "ray.tune.logger.NoopLogger"}
        TD3config['evaluation_interval'] = 1
        TD3config['evaluation_num_episodes'] = 3

        # Env configs
        TD3config['horizon'] = 10 # 每个episode的最长时间步，超过该值时会新起一个episode
        TD3config['callbacks'] = MyCallbacks

        # Model configs
        TD3config['actor_hiddens'] = [5]
        TD3config['actor_hidden_activation'] = 'tanh'
        TD3config['critic_hiddens'] = [5]
        TD3config['critic_hidden_activation'] = 'tanh'
        TD3config['critic_lr'] = 0.001 # Default 0.001
        TD3config['actor_lr'] = 0.001 # Default 0.001
        TD3config['train_batch_size'] = 32
        TD3config['learning_starts'] = 5 # Default 10000
        # Algorithm configs

        TD3config['gamma'] = 0.99  # Default 0.99
        TD3config['exploration_config'] = {
            'type': 'GaussianNoise', 
            'random_timesteps': 5, 
            'stddev': 0.1, 
            'initial_scale': 1.0, 
            'final_scale': 0.001, 
            'scale_timesteps': 5}
        TD3config['timesteps_per_iteration'] = 5
        TD3config['policy_delay'] = 2 # Default 2
        TD3config['target_noise'] = 0.01  # Default 0.2
        TD3config['target_noise_clip'] = 0.05 # Default 0.5
        
        for i in TD3config.keys():
            print(f"{i}: {TD3config[i]}")
        # 3. 创建PPO Trainer实例，传入训练配置和用于交互的env
        # pdb.set_trace()
        trainer = td3.TD3Trainer(config=TD3config, env="MyEnv")
        # 4. 启动三个迭代的训练，可通过train入口跟踪训练过程
        for i in range(3):
            print(f"The training results of {i}th iterations: {trainer.train()}.")

    except Exception as e:
        print("################## Task exit abnormally   ##############")
        print("exception:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    # ray.init(local_mode=True)
    ray.init()
    main()
    ray.shutdown()