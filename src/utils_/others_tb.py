import numpy as np
import random
import torch
import torch.nn as nn
from time import sleep

def epsilon_greedy(env, model, state, epsilon, epsilon_decay):
    '''Exploración vs predicción'''
    random = np.random.uniform(0, 1, size=1)
    if (1 - epsilon) > random:
        action = torch.argmax(model(torch.tensor(state).unsqueeze(0).cuda())).item()
    else:
        action = env.action_space.sample()
    
    epsilon *= epsilon_decay
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
    print('Score: ', score)
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

        #Calculamos el target
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

def training(episode, env, frame_count, model, target_model, epsilon, epsilon_decay,
        state_memory, action_memory, reward_memory, new_state_memory, done_memory,
        batch_size, gamma, score_list, optimizer):    
    for i in range(episode):
        score = 0   
        done = False
        state = env.reset()
        state = np.array(state)
        for u in range(env.spec.max_episode_steps):
            
            action, epsilon = epsilon_greedy(env, model, state, epsilon, epsilon_decay)
            epsilon = max(epsilon, 0.01)
            
            new_state, reward, done, _ = env.step(action)
            new_state = np.array(new_state)
            score += reward

            state_memory.append(state)
            action_memory.append(action)
            reward_memory.append(reward)
            new_state_memory.append(new_state)
            done_memory.append(done)
            
            memory_replay(model, target_model, batch_size, gamma, state_memory, action_memory, 
                    reward_memory, new_state_memory, done_memory, optimizer)
            state = new_state
            frame_count += 1
            
            if done:
                break
        
        # Realizamos cada 10 episodios las siguientes acciones
        if i % 10 == 0:
            torch.save(model.state_dict(), 'modelo_provisional.h5')
            print(f'Episode: {i} --> Score {score}')
            print('epsilon: ', epsilon)
            print('frame_count: ', frame_count)

        score_list.append(score)

def training_various_models():
    for i in range(len()):    
        training(episode, env, frame_count, model, target_model, epsilon, epsilon_decay,
                state_memory, action_memory, reward_memory, new_state_memory, done_memory,
                batch_size, gamma, score_list, optimizer)
