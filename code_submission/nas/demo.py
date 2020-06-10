import torch
import numpy as np
from .controller import GNNController
from .search_space import MacroSearchSpace


def get_reward(actions, gamma=0.99, lambd=0.5):
    rewards = np.zeros_like(actions.shape)
    # TODO: 从actions中构造gnn并计算acc

    # TD(\lambda) return
    batch_size, order = rewards.shape
    for t in range(order):
        base = rewards[:, t:]
        g_t_lambd = np.zeros_like(rewards[:, t], dtype=np.float)
        for step in range(order - t):
            g_t_n = base[:, 0: step + 1]
            gammas = np.power(gamma, np.arange(0, g_t_n.shape[1]))
            g_t_n = np.sum(g_t_n * gammas, axis=1)
            g_t_n *= np.power(lambd, step)
            g_t_lambd += g_t_n
        rewards[:, t] = (1 - lambd) * g_t_lambd \
            + np.power(lambd, order - t) *\
            np.sum(base * np.power(gamma, np.arange(0, base.shape[1])), axis=1)
    return rewards


def train_controller(controller, optimizer, epochs=20, batch_size=1):
    baseline = None
    entropy_history = []
    reward_history = []
    for epoch in range(epochs):
        controller.train()
        optimizer.zero_grad()

        actions, log_probs, entropies = controller.sample(batch_size=batch_size)
        np_entropies = entropies.cpu().numpy()

        rewards = get_reward(actions)
        reward_history.append(rewards)
        entropy_history.append(np_entropies)

        # TODO: loss再考虑
        loss = -log_probs * rewards
        loss = loss.sum()
        loss.backward()
        optimizer.step()
    return controller


if __name__ == '__main__':
    search_space = MacroSearchSpace()
    nas = GNNController(
        search_space.num_tokens,
        hidden_dim=64
    )
    optimizer = torch.optim.Adam(
        nas.parameters(),
        lr=0.005
    )
    nas = train_controller(nas, optimizer)
