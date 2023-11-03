import numpy as np
from torch.utils.data import DataLoader
import torch
import random

import subway_dataset
import training_CNN_learner
import env
import game
import imitation_learner
import expert_policy


# The subway surfer environment
ss_env = env.ImageSubwaySurferEnv()


def get_trained_model(observations, labels, batch_size=32, num_epochs=20):
    train_loader = DataLoader(subway_dataset.SubwayDataset(observations=observations, labels=labels),
                              batch_size=batch_size, shuffle=True)
    return training_CNN_learner.train_learner_given_dataloader(train_loader, num_epochs=num_epochs)


def rollout_trajectories(T, model, expert, beta):
    trajectories = (np.empty((0, 1, 41, 45)), np.empty(0))
    obs, done, j = ss_env.reset(), [False], 0
    while not done[0] or trajectories[0].shape[0] < T:
        # If we have ended prematurely we restart
        if done[0]:
            ss_env.game_live = False
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.no_grad():
            predicted_probabilities = model(torch.from_numpy(obs).to(torch.float32))
            model_action = game.Action(torch.argmax(predicted_probabilities, dim=1).tolist()[0])
        expert_action, _ = expert.predict(obs)
        # Try action of expert with probability beta, otherwise the model
        random_number = random.random()
        action = expert_action if random_number <= beta else model_action
        # Collect trajectory with state and action of expert
        if not done[0] and not np.all(obs == 0):
            trajectories = (np.concatenate((trajectories[0], obs), axis=0),
                            np.concatenate((trajectories[1], expert_action), axis=0))
        # If we have ended prematurely we restart
        # Execute action
        obs, _, done, _ = ss_env.step(action)
        j += 1

    return trajectories


# Rollout using only the learner model for QE-DAgger
def learner_rollout(T, model):
    trajectories = np.empty((0, 1, 41, 45))
    obs, done, j = ss_env.reset(), [False], 0
    while not done[0] or trajectories.shape[0] < T:
        # If we have ended prematurely we restart
        if done[0]:
            ss_env.game_live = False
        if isinstance(obs, tuple):
            obs = obs[0]
        with torch.no_grad():
            predicted_probabilities = model(torch.from_numpy(obs).to(torch.float32))
            model_action = game.Action(torch.argmax(predicted_probabilities, dim=1).tolist()[0])
        # Collect trajectory with state
        trajectories = np.concatenate((trajectories[0], obs), axis=0)
        # Execute action
        obs, _, done, _ = ss_env.step(model_action)
        j += 1

    return trajectories


def DAgger(N=20, T=50):
    # Initializing dataset (observations, acts) which are both np.arrays of the same length
    dataset = (np.zeros((1, 1, 41, 45)), np.zeros(1))
    # Initializing First Model
    cur_model = get_trained_model(dataset[0], dataset[1], num_epochs=1)
    # Get expert policy
    expert = expert_policy.HumanPolicy()
    # Initialize Beta
    beta = 1
    final_beta = 0.3

    # Iterate on rollouts and training experts
    for i in range(N+1):
        print(f"STARTING EPOCH {i+1}/{N} WITH BETA: {beta}")
        # Get the acts of both our expert and the training policy
        dataset_i = rollout_trajectories(T, cur_model, expert, beta)
        # Concatenate dataset
        dataset = tuple(np.concatenate([a, b], axis=0) for a, b in zip(dataset, dataset_i))
        # Train the new model
        cur_model = get_trained_model(dataset[0], dataset[1], num_epochs=20)
        # Get new Beta <- Linear Approach Initially
        beta = max(final_beta, beta - 4*(1 - final_beta)/(3*N))

    return cur_model


class model_wrapper:
    def __init__(self, model_to_wrap):
        self.model = model_to_wrap

    def predict(self, obs):
        with torch.no_grad():
            predicted_probabilities = self.model(torch.from_numpy(obs).to(torch.float32))
            argmax = torch.argmax(predicted_probabilities, dim=1).tolist()[0]
            return game.Action(argmax), None


if __name__ == '__main__':
    #raw_model = DAgger(10)
    #torch.save(raw_model, 'dagger-model')
    raw_model = torch.load('dagger-model')
    wrapped_model = model_wrapper(raw_model)
    imitation_learner.evaluate_learner(wrapped_model, 10)
