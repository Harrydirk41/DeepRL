import torch

import numpy as np
from Memory import Memory


class Agent(object):

    def __init__(self, env, state_dim, action_dim, memory_capacity, reward_gamma=0.99,
                 actor_hidden=32, critic_hidden=32, actor_lr=0.01, critic_lr=0.01, critic_loss="mse", batch_size=100,
                 episode_before_train=100, epsilon_large=0.9, epsilon_small=0.01, epsilon_decay=200, sim_learn_step=30, time_low=0.02, time_high=0.05, cuda = True, entropy = 0.01, grad_norm = 0.5):
        super(Agent, self).__init__(env, state_dim, action_dim, memory_capacity, reward_gamma,
                                    actor_hidden, critic_hidden, actor_lr, critic_lr, critic_loss, batch_size,
                                    episode_before_train, epsilon_large, epsilon_small, epsilon_decay, sim_learn_step, time_low, time_high, cuda, entropy, grad_norm)

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma

        self.actor_hidden = actor_hidden
        self.critic_hidden =critic_hidden
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.epsidoe_before_train = episode_before_train
        self.epsilon_large = epsilon_large
        self.epsilon_small = epsilon_small
        self.epsilon_decay = epsilon_decay
        self.sim_learn_step = sim_learn_step
        
        self.memory = Memory(memory_capacity, batch_size)
        self.cuda = cuda and torch.cuda.is_available()
        
        self.time_low = time_low
        self.time_high = time_high

        self.env_state = self.env.reset()
        self.episode = 0
        self.step = 0
        self.episode_done = False

        self.entropy = entropy
        self.grad_norm = grad_norm
        
    def sim_learn(self):
        pass
    
    def sim_learn_step(self):
        states, actions, rewards, time_step = [],[],[],[]
        for i in range(self.sim_learn_step):
            cur_time_step = np.random.uniform(self.time_low,self.time_high)
            action = self.explore(self.env_state, cur_time_step)
            next_state, reward, done, info = self.env.step(action, cur_time_step)
            final_state = next_state
            self.env_state = next_state

            states.append(self.env_state)
            time_step.append(cur_time_step)
            actions.append(action)
            rewards.append(reward)

            if done:
                self.env_state = self.env.reset()
                break

        next_time_step = np.random.uniform(self.time_low, self.time_high)
        if done:
            final_value = 0
            self.episode +=1
            self.episode_done = True

        else:
            self.episode_done = False
            final_action = self.action(final_state, next_time_step)
            final_value = self.value(final_state, final_action, next_time_step)

        rewards = self.reward_discount(rewards, final_value)
        self.step += 1
        self.memory.push(states,actions,rewards,time_step)

    def reward_discount(self, rewards, values):
        reward = np.zeros_like(rewards)
        add = values
        for i in reversed(range(0, len(rewards))):
            add = add * self.reward_gamma +rewards[i]
            reward[i] = add
        return reward


    def explore(self, state, time_step):
        pass

    def action(self,state, time_step):
        pass

    def value(self,state, action, time_step):
        pass

    def result(self, env, episode = 20):
        rewards = []
        done = 0
        for i in range(episode):
            rewards_i = []
            state = env.reset()
            while (not done):
                time_step = np.random.uniform(self.time_low, self.time_high)
                action = self.action(state, time_step)
                state, reward, done, info = env.step(action, time_step)
                rewards_i.append(reward)

            rewards.append(rewards_i)

        return rewards





        
