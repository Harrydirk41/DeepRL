import torch
from torch import nn
from torch.optim import Adam, RMSprop
from Utils import var2ten,one_hot, entropy

from Agent import Agent
from Actor_Critic import Actor, Critic_Deep

import numpy as np
class A2C(Agent):

    def __init__(self,env, state_dim, action_dim, memory_capacity, reward_gamma=0.99,
                 actor_hidden=32, critic_hidden=32, actor_lr=0.01, critic_lr=0.01, critic_loss="mse", batch_size=100,
                 episode_before_train=100, epsilon_large=0.9, epsilon_small=0.01, epsilon_decay=200, sim_learn_step=30, time_low=0.02, time_high=0.05, cuda = True, entropy = 0.01, grad_norm = 0.5):

        super(A2C, self).__init__(env, state_dim, action_dim, memory_capacity, reward_gamma,
                                    actor_hidden, critic_hidden, actor_lr, critic_lr, critic_loss, batch_size,
                                    episode_before_train, epsilon_large, epsilon_small, epsilon_decay, sim_learn_step, time_low, time_high, cuda, entropy, grad_norm)

        self.actor = Actor(self.state_dim, self.actor_hidden, self.action_dim)

        self.critic = Critic_Deep(self.state_dim, self.action_dim,
                                  self.critic_hidden, 8)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.cuda:
            self.actor.cuda()


    def sim_learn(self):
        super(A2C,self).sim_learn_step()

    def train(self):
        sample = self.memory.sampe()

        states = var2ten(sample.states, self.cuda).view(-1, self.state_dim)
        actions = var2ten(one_hot(sample.action,self.action_dim),self.cuda).view(-1,self.action_dim)
        rewards = var2ten(sample.rewards, self.cuda).view(-1,1)
        time = var2ten(sample.time_step,self.cuda).view(-1,1)

        self.actor_optimizer.zero_grad()
        action_probs = self.actor(states, time)
        entropy_loss = torch.mean(entropy(torch.exp(action_probs)))
        action_probs = torch.sum(action_probs * actions, 1)

        value = self.critic(states, actions, time)
        p_loss = - torch.mean(action_probs * (rewards - value.detach()))
        actor_loss = p_loss - entropy_loss * self.entropy
        actor_loss.backward()
        nn.utils.clip_grad_norm(self.actor.parameters(), self.grad_norm)

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        target = rewards
        critic_loss = nn.MSELoss()(value,target)
        nn.utils.clip_grad_norm(self.critic.parameters(), self.grad_norm)

        self.critic_optimizer.step()

    def softmax_action(self, state, time_step):
        state_input = var2ten([state], self.cuda)
        time_input = var2ten([time_step], self.cuda)
        action = torch.exp(self.actor(state_input,time_input))
        if self.cuda:
            action = action.data.cpu().numpy()[0]
        else:
            action = action.data.numpy()[0]
        return action

    def explore(self, state, time_step):
        action = self.softmax_action(state,time_step)
        epsilon = self.epsilon_small + (self.epsilon_large - self.epsilon_small) * \
                  np.exp(-1. * self.step / self.epsilon_decay)
        factor = np.random.rand()

        if factor < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(action)

        return action

    def action(self,state, time_step):
        action = self.softmax_action(state,time_step)
        action = np.argmax(action)
        return action

    def value(self,state, action, time_step):
        state_input = var2ten(state, self.cuda)
        action_input = var2ten([one_hot(action,self.action_dim)],self.cuda).view(-1,self.action_dim)
        time_input = var2ten([time_step], self.cuda).view(-1,1)
        value = self.critic(state_input, action_input, time_input)
        if self.cuda:
            value = value.data.cpu().numpy()[0]
        else:
            value = value.data.numpy()[0]
        return  value




