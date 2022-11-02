
# coding: utf-8

# In[17]:


#
# This code inspired and encapsulated from Karpathy
# https://gist.github.com/karpathy/d4dee566867f8291f086

import numpy as np
import matplotlib.pyplot as plt


# In[18]:


class simpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate = 1e-1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Parameters tp be trained
        # Xavier initalization
        self.W_xh = np.random.normal(0.0, pow(hidden_dim, -0.5), (hidden_dim, input_dim))
        self.W_hh = np.random.normal(0.0, pow(hidden_dim, -0.5), (hidden_dim, hidden_dim))
        self.W_hy = np.random.normal(0.0, pow(hidden_dim, -0.5), (input_dim, hidden_dim))
        self.b_h = np.zeros((hidden_dim, 1))
        self.b_y = np.zeros((input_dim, 1))
        
        self.mW_xh = 0
        self.mW_hh = 0
        self.mW_hy = 0
        self.mb_h = 0
        self.mb_y = 0
        
        self.vW_xh = 0
        self.vW_hh = 0
        self.vW_hy = 0
        self.vb_h = 0
        self.vb_y = 0
        
    def feedforward(self, input_list, target_list, h_prev):
        x, h, y, p = {}, {}, {}, {}
        h[-1] = np.copy(h_prev) # -1 indicates the last value, but it is used temporally for previous h
        loss = 0
        
        for t in range(len(input_list)):
            x[t] = np.zeros((input_dim, 1))
            x[t][input_list[t]] = 1 # one-hot encode of inpu_list[t]
            h[t] = np.tanh(np.dot(self.W_xh, x[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)
            y[t] = np.dot(self.W_hy, h[t]) + self.b_y
            p[t] = np.exp(y[t])/np.sum(np.exp(y[t])) # list / number
            
            loss += -np.log(p[t][target_list[t],0])
            
        return x, h, y, p, loss, h[len(input_list)-1]
    
    def backprop_thru_time(self, input_list, target_list, x, h, y, p):
        dW_xh, dW_hh, dW_hy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
        dh_next = np.zeros_like(h[0])
        
        for t in reversed(range(len(input_list))):
            dy = np.copy(p[t])
            dy[target_list[t]] -= 1
            dW_hy += np.dot(dy, h[t].T)
            db_y += dy
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_common = (1 - h[t]*h[t]) * dh
            db_h += dh_common
            dW_xh += np.dot(dh_common, x[t].T)
            dW_hh += np.dot(dh_common, h[t-1].T)
            dh_next = np.dot(self.W_hh.T, dh_common)
        
        # as the learning length increases, loss tents to explode
        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)
            
        self.update_weight(dW_xh, dW_hh, dW_hy, db_h, db_y)
         
        return loss, h[len(input_list)-1]
    
    def update_weight(self, dW_xh, dW_hh, dW_hy, db_h, db_y):
        # 
        ## Gradient Descent: lr = 1e-1
        #self.W_xh -= self.learning_rate * dW_xh
        #self.W_hh -= self.learning_rate * dW_hh
        #self.W_hy -= self.learning_rate * dW_hy
        #self.b_h -= self.learning_rate * db_h
        #self.b_y -= self.learning_rate * db_y

        # 
        ## AdaGrad: lr = 1e-1
        #self.vW_xh += dW_xh * dW_xh
        #self.vW_hh += dW_hh * dW_hh
        #self.vW_hy += dW_hy * dW_hy
        #self.vb_h += db_h * db_h
        #self.vb_y += db_y * db_y        
        
        #self.W_xh -= self.learning_rate * dW_xh / np.sqrt(self.vW_xh + 1e-8)
        #self.W_hh -= self.learning_rate * dW_hh / np.sqrt(self.vW_hh + 1e-8)
        #self.W_hy -= self.learning_rate * dW_hy / np.sqrt(self.vW_hy + 1e-8)
        #self.b_h -= self.learning_rate * db_h / np.sqrt(self.vb_h + 1e-8)
        #self.b_y -= self.learning_rate * db_y / np.sqrt(self.vb_y + 1e-8)
        
        # 
        # Adam: lr = 1e-3
        self.mW_xh = 0.9*self.mW_xh + 0.1*dW_xh
        self.mW_hh = 0.9*self.mW_hh + 0.1*dW_hh
        self.mW_hy = 0.9*self.mW_hy +0.1*dW_hy
        self.mb_h = 0.9*self.mb_h + 0.1*db_h
        self.mb_y = 0.9*self.mb_y + 0.1*db_y
        
        self.vW_xh = 0.999*self.vW_xh + 0.001*(dW_xh * dW_xh)
        self.vW_hh = 0.999*self.vW_hh + 0.001*(dW_hh * dW_hh)
        self.vW_hy = 0.999*self.vW_hy + 0.001*(dW_hy * dW_hy)
        self.vb_h = 0.999*self.vb_h + 0.001*(db_h * db_h)
        self.vb_y = 0.999*self.vb_y + 0.001*(db_y * db_y)
        
        self.W_xh -= self.learning_rate*((10*self.mW_xh)/np.sqrt(1000*self.vW_xh+1e-8))
        self.W_hh -= self.learning_rate*((10*self.mW_hh)/np.sqrt(1000*self.vW_hh+1e-8))
        self.W_hy -= self.learning_rate*((10*self.mW_hy)/np.sqrt(1000*self.vW_hy+1e-8))
        self.b_h -= self.learning_rate*((10*self.mb_h)/np.sqrt(1000*self.vb_h+1e-8))
        self.b_y -= self.learning_rate*((10*self.mb_y)/np.sqrt(1000*self.vb_y+1e-8))        
        
    def sample(self, h, seed_ix, n):
        x = np.zeros((self.input_dim, 1))
        x[seed_ix] = 1
        ixes = []
  
        for t in range(n):
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.input_dim), p=p.ravel())
            x = np.zeros((self.input_dim, 1))
            x[ix] = 1
            ixes.append(ix)
            
        return ixes


# In[19]:


data = open('deep learning nature 2015.txt', 'r').read()

s = set(data)   # Remove overapped characters
characters = list(s) # Add the non-overlapped characters into list

data_size = len(data)
input_dim = len(characters)
hidden_dim = 100
learning_length = 25

char_to_num = {ch:i for i, ch in enumerate(characters)}
num_to_char = {i:ch for i, ch in enumerate(characters)}

rnn = simpleRNN(input_dim, hidden_dim, input_dim, 1e-3)

pointer, iteration = 0, 0
h_prev = np.zeros((hidden_dim, 1))
lowpass_filtered_loss = -np.log(1.0/input_dim)*learning_length
max_iteration = 30000
loss_list = list()
lowpass_filtered_loss_list = list()

while True:
    if pointer+learning_length >= len(data):
        h_prev = np.zeros((hidden_dim,1)) # reset RNN memory
        pointer = 0 # go from start of data
    
    input_list = [char_to_num[c] for c in data[pointer:pointer+learning_length]]
    target_list = [char_to_num[c] for c in data[pointer+1:pointer+learning_length+1]]
    
    x, h, y, p, loss, h_prev = rnn.feedforward(input_list, target_list, h_prev)
    rnn.backprop_thru_time(input_list, target_list, x, h, y, p)
    lowpass_filtered_loss = lowpass_filtered_loss * 0.99 + loss * 0.01
    
    loss_list.append(loss)
    lowpass_filtered_loss_list.append(lowpass_filtered_loss)
    
    pointer += learning_length
    iteration += 1
    
    if iteration % 100 == 0:
        sample_ix = rnn.sample(h_prev, input_list[0], 200)
        txt = ''.join(num_to_char[ix] for ix in sample_ix)
        print(iteration, ": loss: ", lowpass_filtered_loss, '----\n %s \n----' % (txt, ))


# In[21]:


x_t = np.arange(1, max_iteration, 1)

plt.figure(1)
plt.plot(x_t, loss_list[1:max_iteration], 'k')

plt.figure(2)
plt.plot(x_t, lowpass_filtered_loss_list[1:max_iteration], 'k')

