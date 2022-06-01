import numpy as np
import matplotlib.pyplot as plt
from environment.cart_pole import CartPoleEnv
from gym_deep.time_limit import TimeLimit
from A2C import A2C

MEMORY_CAPACITY = 50000
REWARD_GAMMA = 0.99
ACTOR_LR = 0.01
CRITIC_LR = 0.01
BATCH_SIZE = 100
EPSILON_LARGE = 0.99
EPSILON_SMALL = 0.01
EPSILON_DECAY = 500
SIM_LEARN_STEP = 30

EPISODE_BEFORE_TRAIN = 100


VAL_EPISODE = 5000
VAL_STEP = 10

def run(agent_name):
    env_train = CartPoleEnv()
    env_train = TimeLimit(env_train, max_episode_steps=200)

    env_val = CartPoleEnv()
    env_val = TimeLimit(env_val, max_episode_steps=200)

    state_dim = env_train.observation_space.shape[0]
    if len(env_train.action_space.shape) > 1:
        action_dim = env_train.action_space.shape[0]
    else:
        action_dim = env_train.action_space.n

    if agent_name == "A2C":
        agent = A2C(env=env_train,state_dim=state_dim,action_dim=action_dim,memory_capacity=MEMORY_CAPACITY,
                    actor_lr=ACTOR_LR,critic_lr=CRITIC_LR,episode_before_train=EPISODE_BEFORE_TRAIN,
                    epsilon_large=EPSILON_LARGE, epsilon_small=EPSILON_SMALL,epsilon_decay=EPSILON_DECAY,
                    sim_learn_step=SIM_LEARN_STEP)

    val_episodes = []
    val_rewards = []
    while agent.episode < VAL_EPISODE:
        agent.sim_learn()
        if agent.episode >= EPISODE_BEFORE_TRAIN:
            agent.train()
        if agent.episode_done and (agent.episode + 1) % 100 == 0:
            rewards = agent.result(env_val, VAL_STEP)
            rewards_sum = [np.sum(np.array(i), 0 ) for i in rewards]
            mean_rewards = np.mean(np.array(rewards_sum), 0 )
            print("Episode %d Average Reward %.2f" %(agent.episode + 1, mean_rewards))
            val_episodes.append(agent.episode + 1)
            val_rewards.append(mean_rewards)

    plt.figure()
    plt.plot(val_episodes, val_episodes)
    plt.xlabel("Episodes")
    plt.ylabel("average Reward")
    plt.savefig("./cartpole.png")


