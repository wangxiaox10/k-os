# -*- coding: utf-8 -*-
from Pdistribution import *

k=2
K=20
Pdist = Pdistribution(k, K)

for i in range(1,K+1):
    print Pdist.chooseOneDistribution(i)
    
