import numpy as np

a = np.array([
    [1, 3, 5],
    [5, 4, 1]
]).astype(np.float32)

b = np.array([
    [6, 8, 11, 5],
    [3, 5, 8, 12],
    [9, 1, 8, 7]
]).astype(np.float32)

c = np.dot(a, b)

print(a)
print(b)
print(c)

#print(np.dot(b.T, a.T))
#print(b.T)
