'''
This file is agent for the part of first EXO proj.
And this work submitted to KMMS spring conference.
this simulation result will 
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras 
from collections import deque
import numpy as np


def rand_agent(env, state):
    ois_1 = state[0]
    Dis = state[1] 
    Eus = state[2] 
    
    alarm = True
    while alarm:
        FUmax = env.FU[1] * 0.9
        FUmin = env.FU[0] 
        FImax = env.FI[1] * 0.9 
        FImin = env.FI[0]

        fus = np.random.rand(env.N) 
        ois = np.random.rand(env.N)

        fus = np.random.rand(env.N) * (FUmax-FUmin) + FUmin
        fli = np.random.rand(env.N) * (FImax-FImin) + FImin
        ois = np.random.rand(env.N)

        tlis = env.local(fli, ois_1, Dis)[1]
        Eexe ,texe = env.computing(fus, ois_1, Dis)[1]

        alarm = bool((tlis < env.ts) | (texe < env.ts) & (Eexe < Eus))
    
    action = [fus, ois]
    return action 

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPG():
    def __init__(self, env, buffer_capacity=1000, learning_rate=0.01, batch_size=32, discount=0.9):
        self.env = env
        self.state_dim = 3 * env.N + 2 # why add 2? -> because it is vector
        self.action_dim = 3 * env.N + 2
        self.learning_rate = learning_rate
        self.critic_learning_rate = 2*learning_rate
        self.gamma = discount
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.train_start = 10 
        self.xi = 0.001 # for update target networks
        self.kn_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

        #---- Replay memory --------------------------------------------------
        self.buffer = deque(maxlen = self.buffer_capacity)
        
        #---- Creat a noise process ------------------------------------------
        self.mean = 0.0
        self.std = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.noise = OUActionNoise(mean = self.mean*np.ones(self.action_dim), 
                             std_deviation = self.std*np.ones(self.action_dim))
        
        #---- Create actor and critic ----------------------------------------
        self.actor = self.get_actor()
        self.target_actor = self.get_actor()
        self.critic = self.get_critic()
        self.target_critic = self.get_critic()
        
        # Make the weights equal initially
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_learning_rate)

    def record(self, obs_tuple):
        # Saves experience tuple (s,a,r,s') in the replay memory
        self.buffer.append(obs_tuple)
    
    def get_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        hidden = layers.Dense(128, activation="relu")(inputs)
        hidden = layers.Dense(128, activation="relu")(hidden)
        
        # action = [ut, fnt, pnt, bnt]
        action = layers.Dense(self.action_dim, activation="sigmoid", kernel_initializer=self.kn_init)(hidden)
        
        # Outputs actions
        model = keras.Model(inputs=inputs, outputs=action)
        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.state_dim,))
        state_out = layers.Dense(128, activation="relu")(state_input)
        state_out = layers.Dense(128, activation="relu")(state_out)
        
        # Action as input
        action_input = layers.Input(shape=(self.action_dim,))
        action_out = layers.Dense(128, activation="relu")(action_input)
        action_out = layers.Dense(128, activation="relu")(action_out)
    
        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])
        
        hidden = layers.Dense(256, activation="relu")(concat)
        hidden = layers.Dense(256, activation="relu")(hidden)
        Qvalue = layers.Dense(1)(hidden)
        
        # Outputs Q-value for given state-action
        model = keras.Model(inputs=[state_input, action_input], outputs=Qvalue)
        return model
   
    def convert_to_vector(self, var_in):
        # Convert a state into a vector for Tensorflow process
        out = np.empty(0)
        for var in var_in:
            out = np.append(out, np.reshape(var, (1,-1)))
        return out

    def policy(self, state, scheme='Proposed'):
        # Return an action sampled from the actor DNN plus some noise for exploration
        # Convert the state into a vector, then to a tensor
        state_vector = self.convert_to_vector(state)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state_vector),0)
        
        # Sample action from the actor, and add noise to the sampled actions
        sampled_action = tf.squeeze(self.actor(tf_state))
        sampled_action = sampled_action.numpy() + self.epsilon * self.noise()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

         # Make sure actions are within bounds
        action = np.clip(sampled_action,0,1)
        
        # actions = [ut, fnt, pnt, bnt]

        # will change below
        fus= action[0]
        fli = action[1:self.env.N + 2]
        ois = action[self.env.N + 2:]
        
        if scheme == 'Uniform':
            fus = 0.9*np.random.rand(1)
            fli = 0.9*np.random.rand(self.env.N)
            ois = np.random.rand(0,1,self.env.N)
        
        if scheme == 'Random':
            fus = 0.9*np.random.rand(1)
            fli = np.random.rand(1,5,self.env.N)
            ois = np.random.rand(0,1,self.env.N)
        
        return [fus, fli, ois]

     # Use tf.function to speed up blocks of code that contain many small TensorFlow operations.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, state_next_batch):
        # Train the actor and the critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(state_next_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic([state_next_batch, target_actions], training=True)
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
    
    @tf.function
    def update_target(self):
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(b * self.xi + a * (1 - self.xi))
        
        for (c, d) in zip(self.target_critic.variables, self.critic.variables):
            c.assign(d * self.xi + c * (1 - self.xi))
    
    def update_model(self):
        # Select random samples from the buffer to train the actor and the critic
        if len(self.buffer) < self.train_start:
            return
        
        indices = np.random.choice(len(self.buffer), self.batch_size)
        state_batch, action_batch, reward_batch, state_next_batch = [], [], [], []
        for i in indices:
            state_batch.append(self.convert_to_vector(self.buffer[i][0]))
            action_batch.append(self.convert_to_vector(self.buffer[i][1]))
            reward_batch.append(self.buffer[i][2])
            state_next_batch.append(self.convert_to_vector(self.buffer[i][3]))
        
        # Convert to tensors
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        reward_batch = tf.convert_to_tensor(reward_batch)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        state_next_batch = tf.convert_to_tensor(state_next_batch)
        
        # Update parameters
        self.update(state_batch, action_batch, reward_batch, state_next_batch)
