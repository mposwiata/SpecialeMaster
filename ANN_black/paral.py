from multiprocess import Pool
from time import sleep
import time

def f(x):
    sleep(1)
    return x + 1

def g(s, x):
    sleep(s)
    return x + 1

# [f(i) for i in range(5)]  # slow

pool = Pool(4)
starttime = time.time()
# res = pool.map(f, range(12))
# res2 = pool.map(lambda a: g(1, a), range(12))
res3 = pool.starmap(g, [[1,1], [1, 2], [1,3], [1,4]])
print(time.time() - starttime)



