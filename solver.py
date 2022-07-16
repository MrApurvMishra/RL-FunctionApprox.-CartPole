'''
    APS1080 - Introduction to Reinforcement Learning
    Assignment 4
    Environment used: CART-POLE V1
    Author: Apurv Mishra
'''


# import libraries
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# action for epsilon greedy action
def epsilon_greedy_action(epsilon, Q, obs):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        estimate = Q(tf.expand_dims(obs, axis=0))
        action = np.argmax(estimate)
    return action


# function to test the policy for trial runs
def check_policy(env, model, max_runs=100):
    rewards_list = []
    for i in range(max_runs):
        obs     = env.reset()
        rewards = 0
        done    = False
        while not done:
            estimate = model(tf.expand_dims(obs, axis=0))[0]
            action   = np.argmax(estimate)
            obs, reward, done, _ = env.step(action)
            rewards += reward
        rewards_list.append(rewards)
        print("Checking the policy with trial runs...", 100 * i/max_runs, "%")
    return rewards_list


# build a NN model - MLP-5
def build_model(input_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(output_dim, activation="linear"))
    model.summary()
    return model


# initialize the environment
env = gym.make('CartPole-v1')  # define the environment

# hyperparameters
alpha   = 0.001
gamma   = 0.9
epsilon = 0.5

# states and actions
n_states  = 4
n_actions = 2

# state-action value function
model = build_model(n_states, n_actions)

# optimizer and per-prediction error
optimizer = keras.optimizers.SGD(learning_rate=alpha)
loss_fn   = keras.losses.MeanSquaredError()

# start iterations
converged   = False
n_episode   = 0
rewards     = 0
reward_list = []
pp_err_list = []
while not converged:
    error_avg = 0
    n_episode += 1
    done = False

    # generating state and action pair
    rewards = 0
    curr_obs = env.reset()
    curr_action = env.action_space.sample()

    # run through the episode
    while not done:
        next_obs, reward, done, _ = env.step(curr_action)
        rewards += reward
        if done:
            with tf.GradientTape() as tape:
                prediction = model(tf.expand_dims(curr_obs, axis=0), training=True)[0]
                q_estimate = prediction[curr_action]
                ppError    = loss_fn(tf.expand_dims(reward, axis=0), tf.expand_dims(q_estimate, axis=0))
        else:
            next_action = epsilon_greedy_action(epsilon, model, next_obs)
            with tf.GradientTape() as tape:
                next_prediction = model(tf.expand_dims(next_obs, axis=0), training=True)[0]
                next_q_estimate = reward + gamma * next_prediction[next_action]
                curr_prediction = model(tf.expand_dims(curr_obs, axis=0), training=True)[0]
                curr_q_estimate = curr_prediction[curr_action]
                ppError         = loss_fn(tf.expand_dims(next_q_estimate, axis=0), tf.expand_dims(curr_q_estimate, axis=0))

            # updating current state and action
            curr_obs    = next_obs
            curr_action = next_action

        # updating weights of NN
        grads = tape.gradient(ppError, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # calculate average error for each episode
        error_avg = (error_avg + ppError) / 2

    # print average error for each episode
    print("Episode", n_episode, ": rewards =", rewards, ", average error = %f" % float(error_avg))

    # save rewards and per-prediction errors
    reward_list.append(rewards)
    pp_err_list.append(error_avg)

    # check for convergence
    if n_episode > 10000:
        converged = True

    # reduce exploration
    if n_episode % 1000 == 0 and epsilon > 0.1:
        d = n_episode/100
        epsilon -= 2/d

# plot the rewards
rewards_list = check_policy(env, model)
smoothened_rewards = savgol_filter(reward_list, 55, 3)
plt.figure()
plt.plot(smoothened_rewards)
plt.xlabel("number of episodes")
plt.ylabel("total rewards")
plt.title("Rewards per episode while training")

# plot the per-prediction errorr
smoothened_errors = savgol_filter(pp_err_list, 501, 3)
plt.figure()
plt.plot(smoothened_errors)
plt.xlabel("number of episodes")
plt.ylabel("total rewards")
plt.title("PP error per episode while training")

# print the details of the complete execution output
print("For semi-gradient SARSA:")
print("Total episodes =", n_episode, " with average steps =", np.mean(rewards_list))
