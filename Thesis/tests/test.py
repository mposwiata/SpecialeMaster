import numpy as np
import tikzplotlib
import matplotlib.pyplot as plt

def f(x):
    return 2 * np.sin(x) + 4 * np.cos(x / 2)

x = np.linspace(start = 0, stop = 12.5, num = 100)

y = f(x)

x_saddel = 3 * np.pi
y_saddel = f(x_saddel)

x_min = 5 * np.pi / 3
y_min = f(x_min)

x_rand = 3
y_rand = f(x_rand)

x_rand2 = 3.5
y_rand2 = f(x_rand2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.plot(x_saddel, y_saddel, 'ko') 
ax.plot(x_min, y_min, 'ko')
ax.plot(x_rand, y_rand, 'ko')
ax.plot(x_rand2, y_rand2, 'ko')
ax.set_ylabel("E(w)", rotation="horizontal", labelpad=15)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.annotate("ws", (x_saddel, y_saddel), xytext = (0,4), textcoords="offset points")
ax.annotate("wm", (x_min, y_min), xytext = (0,-10), textcoords="offset points")
ax.annotate("wr", (x_rand, y_rand), xytext = (0,4), textcoords="offset points")
ax.annotate("wr", (x_rand2, y_rand2), xytext = (0,4), textcoords="offset points")

tikzplotlib.save("error.tex")