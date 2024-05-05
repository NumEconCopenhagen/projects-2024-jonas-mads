# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:38:07 2024

@author: jonas
"""

from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        return (x1A ** self.par.alpha) * (x2A ** (1-self.par.alpha))

    def utility_B(self,x1B,x2B):
        return (x1B ** self.par.beta) * (x2B ** (1-self.par.beta))

    def demand_A(self,p1):
        x1A = self.par.alpha * (p1 * self.par.w1A + 1 * self.par.w2A) / (p1)
        x2A = (1 - self.par.alpha) * (p1 * self.par.w1A + 1 * self.par.w2A) / 1
        
        return x1A, x2A

    def demand_B(self,p1):
        x1B = self.par.beta * (p1 * self.par.w1B + 1 * self.par.w2B) / (p1)
        x2B = (1 - self.par.beta) * (p1 * self.par.w1B + 1 * self.par.w2B) / 1
        
        return x1B, x2B

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def market_clearing_new_endowments(self,p1,w1A,w2A):
        
        w1B = 1 - w1A
        w2B = 1 - w2A
        
        x1A = self.par.alpha * (p1 * w1A + 1 * w2A) / (p1)
        x2A = (1 - self.par.alpha) * (p1 * w1A + 1 * w2A) / 1
        
        x1B = self.par.beta * (p1 * w1B + 1 * w2B) / (p1)
        x2B = (1 - self.par.beta) * (p1 * w1B + 1 * w2B) / 1
        
        eps1 = x1A - w1A + x1B - (1 - w1A)
        eps2 = x2A - w2A + x2B - (1 - w2A)

        return eps1,eps2
    
    
