import numpy as np
import time

import pyhack

N = 500
a = np.random.randn(N, N)
b = np.random.randn(N, N)

start = time.time()
c1 = np.dot(a, b)
dur = int((time.time() - start) * 1000)
print('Duration: {}ms'.format(dur))

pyhack.enable_wrapper()

start = time.time()
c2 = np.dot(a, b)
dur = int((time.time() - start) * 1000)
print('Duration: {}ms'.format(dur))

print('Difference = {}'.format(np.linalg.norm(c1 - c2)))
