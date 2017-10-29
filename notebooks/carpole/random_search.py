
# coding: utf-8

# In[1]:


from __future__ import print_function, division
from builtins import range
import gym
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


# In[13]:


def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    
    while not done and t < 200:  #< 200 #<10000
        env.render()
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
            
    return t


# In[14]:


def play_multiple_episodes(env, T, params):
    episode_lengths = np.empty(T)
    
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
        
    avg_length = episode_lengths.mean()
    print("Avg lenght:", avg_length)
    return avg_length


# In[15]:


def random_search(env):
    episode_lenghts = []
    best = 0
    params = None
    for t in xrange(100):
        new_params = np.random.random(4)*2 - 1
        avg_lenght = play_multiple_episodes(env, 100, new_params)
        episode_lenghts.append(avg_lenght)
        
        if avg_lenght > best:
            params = new_params
            best = avg_lenght
    return episode_lenghts, params


# In[16]:


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lenghts, params = random_search(env)
    plt.plot(episode_lenghts)
    plt.show()
    
    # play a final set of episodes
    print("***Final run with final weights***")
    play_multiple_episodes(env, 100, params)

