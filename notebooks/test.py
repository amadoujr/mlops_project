import matplotlib.pyplot as plt
import numpy as np

objective = lambda x: (x-3)**2 + 2


x = np.linspace(-10, 10, 100)
y = objective(x)

fig = plt.figure()
plt.plot(x, y)
plt.show()
