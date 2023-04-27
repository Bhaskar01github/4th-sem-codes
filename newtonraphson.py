import numpy as np
import math
#Tabular Method for function f(x) in interval [0,1]
def f(x):
    return math.exp(x)-5
table = np.arange(1,2+0.1,0.1)   # array starts from 0 with gap of 0.1 till 1
print(table)

for i in range(len(table)-1):
    if f(table[i])*f(table[i+1]) <= 0:           
        print(f"Root lies between {table[i]} and {table[i+1]}")




import numpy as np
import math
def f(x):
  return math.exp(x)-5
def df(x):
  return math.exp(x)
a=float(input("enter the intital guess of the root :"))
i=abs(f(a)/df(a))
x=a
while i>0.001:
    x=x-i
    i=abs(f(x)/df(x))
print("%0.3f"%x)
