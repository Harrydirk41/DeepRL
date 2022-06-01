import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, state_dim, hidden_size_1, hidden_size_2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def __call__(self, time):
        out = nn.functional.relu(self.fc1(time))
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Actor(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + 1, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def __call__(self, state, time_step):
        out = nn.functional.relu(self.fc1(state))
        out = torch.cat([out, time_step], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)

        return out


class Critic(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = torch.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Critic_Deep(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size, output_size):
        super(Critic_Deep, self).__init__()
        self.branch = Critic(state_dim, action_dim, hidden_size, output_size)
        self.trunk = MLP(1, hidden_size, hidden_size, output_size)

    def __call__(self, state, action, time_step):
        B = self.branch(state, action)
        T = self.trunk(time_step)
        sum = torch.sum(B * T)
        return sum


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size,
                 actor_output_act, critic_output_size=1):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val
