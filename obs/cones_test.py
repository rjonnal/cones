import numpy as np
import cones
import csv

x = np.random.rand(10)
y = np.random.rand(10)
labs = ['s','l/m']

z = np.random.rand(10)

z=np.round(z).astype(np.int)
labels = [labs[zidx] for zidx in z]
cs = cones.ConeSet()
for xx,yy,label in zip(x,y,labels):
    cs.append(cones.Cone(xx,yy,label))

cs.to_file('test.csv')

cs2 = cones.ConeSet()
cs2.from_file('test.csv')

print cs
print cs2

cs3 = cs + cs2

print cs3

    
    
