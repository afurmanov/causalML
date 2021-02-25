# 6.1

import torch
import pyro
from pyro.distributions import Categorical
import torch
from collections import Counter

pyro.set_rng_seed(101)

prob_A = torch.tensor([0.36, 0.16, 0.48]) # A=adult, A=old, A=young

prob_S = torch.tensor([0.55, 0.45]) #S=F, S=M

prob_E = torch.tensor([
    [[0.64, 0.36], #E=high|S=F,A=adult, E=uni|S=F,A=adult
    [0.84, 0.16],  #E=high|S=F,A=old,   E=uni|S=F,A=old
    [0.16, 0.84]], #E=high|S=F,A=young, E=uni|S=F,A=young

    [[0.72, 0.28], #E=high|S=M,A=adult, E=uni|S=M,A=adult
    [0.89, 0.11],  #E=high|S=M,A=old,   E=uni|S=M,A=old
    [0.81, 0.19]], #E=high|S=M,A=young, E=uni|S=M,A=young
    ])

prob_O = torch.tensor([
    [0.98, 0.02], #O=emp|E=high,  O=self|E=high
    [0.97, 0.03]  #O=emp|E=uni,   O=self|E=uni
    ])


prob_R = torch.tensor([
    [0.72, 0.28], #R=big|E=high, R=small|E=high
    [0.94, 0.06]  #R=big|E=uni,  R=small|E=uni
    ])

prob_T = torch.tensor([
    [[0.71, 0.14, 0.15], #T=car|R=big,O=emp, T=other|R=big,O=emp, T=train|R=big,O=emp
    [0.68, 0.16, 0.16]], #T=car|R=big,O=self, T=other|R=big,O=self, T=train|R=big,O=self

    [[0.55, 0.08, 0.37], #T=car|R=small,O=emp, T=other|R=small,O=emp, T=train|R=small,O=emp
    [0.73, 0.25, 0.02]], #T=car|R=small,O=self, T=other|R=small,O=self, T=train|R=small,O=self
    ])

def transportation():
    S=pyro.sample("S", Categorical(probs=prob_S))
    A=pyro.sample("A", Categorical(probs=prob_A))
    E=pyro.sample("E", Categorical(probs=prob_E[S][A]))
    O=pyro.sample("O", Categorical(probs=prob_O[E]))
    R=pyro.sample("R", Categorical(probs=prob_R[E]))
    T=pyro.sample("T", Categorical(probs=prob_T[R][O]))
    return A, R, E, O, S, T

# 6.2.a

samples_total = 10000
condtioned_on_E_uni = pyro.condition(transportation, data={"E": torch.tensor(1)})
samples_cond = [int(condtioned_on_E_uni()[5]) for i in range (samples_total)]

histogram_cond = Counter(samples_cond)
for key in histogram_cond:
    histogram_cond[key] /= samples_total
    
print(f'marginal distribution of T given E=uni: {histogram_cond}')
# marginal distribution of T given E=uni: Counter({0: 0.689, 2: 0.164, 1: 0.147})
# means of travel: Car – 69%, train - 15%, other - 16%

