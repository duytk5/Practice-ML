import numpy as np

xx = np.zeros(28*28);
f = open('in.txt', 'r')
str= f.read().split()
list = map(float, str)
d = 0
for y in list:
    xx[d] = y
    d+=1
x_input =  np.array([xx])

print(x_input)
