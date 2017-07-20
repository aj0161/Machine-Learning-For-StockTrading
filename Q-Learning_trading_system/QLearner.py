"""

"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions                                   # number of actions
        self.num_states = num_states                                     # number of states 
        self.s = 0                                                       # states - a point eg: (1,2), (2,2)
        self.a = 0                                                       #Action[east, west, south, north]
        self.t = np.zeros([num_states, num_actions, num_states])         #Transition function T[s,a,s']      
        self.alpha=alpha                                                 #learning rate
        self.gamma=gamma                                                 #discount rate
        self.rar=rar                                                     #random action rate
        self.radr=radr                                                   #random action decay rate
        self.dyna=dyna                                                   #dyna

        #initialize Q[] with uniform random values between -1.0 and 1.0
        self.Q = np.random.uniform(-1.0, 1.0, [num_states,num_actions])  #Q table

        #expected reward for s,a
        self.Exp_Reward = (np.ones([self.num_states, self.num_actions])*-1.0)

        #set to 0.00001 to prevent from divide by zero exception
        self.TC = (np.ones([self.num_states, self.num_actions, self.num_states]) * .00001)

    def author(self):
        return 'ajoshi319'

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action

        Note from Piazza:
        - choose a random action sometimes (based on rar). 
        - choose a random action or argmax Q[s] as in my query(). 
        """
        self.s = s
        random_num = self.get_rand_value() # random number between 0 to 1
        action = self.get_action(random_num, s)
        self.a = action

        if self.verbose: 
            print "s =", s,"a =",action

        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        self.compute_Q(s_prime,r)
        # If the random number is > than our probability,
            #-use policy, 
            #-otherwise, pick a random action

        random_num = self.get_rand_value()
        action =  self.get_action(random_num, s_prime)

        #select random action with decay rate
        self.rar = self.rar * self.radr

        if (self.dyna > 0):
            # Update T'[s,a,s'] and R'[s,a]
            self.update_Dyna_model( s_prime, r)

        # update action and state
        self.s = s_prime
        self.a = action

        #hallucination process
        if (self.dyna > 0):
            self.compute_Dyna_model(s_prime, r)  

        if self.verbose: 
            print "s =", s_prime, "a =", action, "r =", r

        return action

    def get_rand_value(self):
        return np.random.random()

    def get_action(self, rand_num, state):
        if(rand_num > self.rar):
            return np.argmax(self.Q[state])   #choose action = Q[s] where Q is a table and s is state
        else:
            return rand.randint(0, self.num_actions-1)

    def compute_Q(self, s_prime ,r):
        self.Q[self.s, self.a]= (1-self.alpha) * self.Q[self.s,self.a] + self.alpha * (r + self.gamma * self.Q[s_prime,np.argmax(self.Q[s_prime])])

    def update_Dyna_model(self, s_prime, r):
        # increment by one each of location in TC matrix
        self.TC[self.s, self.a, s_prime] += self.TC[self.s, self.a, s_prime]

        # update expected reward
        self.Exp_Reward[self.s, self.a] = (1 - self.alpha) * self.Exp_Reward[self.s, self.a] + self.alpha * r

        # how many times that tranisition occurs (TC[s,a,s_prime])
        numerator = self.TC[self.s, self.a, s_prime]

        #total number of times that we are in state s and action a (summation of all s_prime)
        denominator = np.sum(self.TC[self.s, self.a, :])

        #compute the probability s_prime given s and a. P[s_prime|s,a]
        self.t[self.s, self.a, s_prime] = numerator / denominator
           
        # normalize T table
        self.t[self.s, self.a, :] = self.t[self.s, self.a, :] /self.t[self.s, self.a, :].sum()

    def compute_Dyna_model(self, s_prime, r):
        s_dyna, a_dyna = np.nonzero(np.sum(self.t, axis=2))
        rand_pick = np.random.choice(len(s_dyna), self.dyna)

        for n in range(0, len(rand_pick)):
             s_prime_Hallu = np.random.choice(range(0, self.num_states), p=self.t[s_dyna[rand_pick[n]], a_dyna[rand_pick[n]], :])
             r_Hallu = self.Exp_Reward[s_dyna[rand_pick[n]], a_dyna[rand_pick[n]]]
             self.Q[s_dyna[rand_pick[n]], a_dyna[rand_pick[n]]] = (1 - self.alpha) * self.Q[s_dyna[rand_pick[n]], a_dyna[rand_pick[n]]] + self.alpha * (r_Hallu + self.gamma * self.Q[s_prime_Hallu, np.argmax(self.Q[s_prime_Hallu])]) 

if __name__=="__main__":
    print "..."