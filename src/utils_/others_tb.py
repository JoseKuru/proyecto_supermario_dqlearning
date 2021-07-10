import numpy as np
import random
import torch
from time import sleep

def epsilon_greedy(env, model, state, epsilon):
    '''Exploración vs predicción'''
    random = np.random.uniform(0, 1, size=1)
    if (1 - epsilon) > random:
        action = model.predict(state)
    else:
        action = env.action_space.sample()
    
    epsilon *= 0.99
    epsilon = max(epsilon, 0.01)

    return action, epsilon

def play_random(env, episodes):    
    for i in range(episodes):
        env.reset()
        done = False
        score = 0
        while not done:
            action = random.choice(np.arange(0, 5))
            state, reward, done, _ = env.step(action)
            score += reward
        print(f'Episode: {i + 1} -> Score: {score}')

def play(env, model):    
    state = np.array(env.reset())
    done = False
    score = 0
    while not done:
        sleep(0.1)
        env.render(mode='human')
        action = torch.argmax(model(torch.tensor(state).unsqueeze(0))).item()
        new_state, reward, done, _ = env.step(action)
        state = new_state
        score += reward
    env.close()

def memory_replay(model, target_model, batch_size, gamma, state_memory, action_memory, reward_memory, 
        new_state_memory, done_memory, optimizer):

    if len(state_memory) < batch_size: 
        return None
    else:
        #Cogemos un sample de los datos que tenemos
        idx = random.choices(range(len(reward_memory)), k=batch_size)

        STATE = torch.tensor(np.array(state_memory)[idx]).cuda()
        ACTION = torch.tensor(np.array(action_memory)[idx]).cuda()
        REWARD = torch.tensor(np.array(reward_memory)[idx]).cuda()
        STATE2 = torch.tensor(np.array(new_state_memory)[idx]).cuda()
        DONE = torch.ByteTensor(np.array(done_memory)[idx]).cuda()

        #Primero calculamos cuales son los valores Q de cada una de las observaciones 
        state_action_values = model(STATE).gather(1, ACTION.unsqueeze(-1)).squeeze(-1)


        next_state_values = target_model(STATE2).max(1)[0]
        next_state_values[DONE] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values=(next_state_values * gamma) + REWARD
        expected_state_action_values = expected_state_action_values.float()
        
        optimizer.zero_grad()
        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values).cuda()
        
        # Descenso de gradiente y backpropagation
        loss_t.backward()
        optimizer.step()