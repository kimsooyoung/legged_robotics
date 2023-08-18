import matplotlib.pyplot as plt
import bezier
import numpy as np

# create curve passing points
# (0.0, 0.0) (0.5, 1.0) (1.0, 0.0)
nodes1 = np.asfortranarray([
    [0.0, 0.5, 1.0],
    [0.0, 1.0, 0.0],
])
nodes2 = np.asfortranarray([
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.0, 2.0, -2.0, 2.0, 0.0],
])
nodes3 = np.asfortranarray([
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [0.0, 0.0, 1.0, 1.0, 1.0],
])

# second order bezier curve
# order should be (# of points - 1)
curve1 = bezier.Curve(nodes1, degree=2)
curve2 = bezier.Curve(nodes2, degree=4)
cuirve3 = bezier.Curve(nodes3, degree=4)

N = 100
t_span = np.linspace(0.0, 1.0, N)
result1 = np.zeros((2, N))
result2 = np.zeros((2, N))
result3 = np.zeros((2, N))

for i, t in enumerate(t_span):
    result1[0, i] = curve1.evaluate(t)[0, 0]
    result1[1, i] = curve1.evaluate(t)[1, 0]

    result2[0, i] = curve2.evaluate(t)[0, 0]
    result2[1, i] = curve2.evaluate(t)[1, 0]

    result3[0, i] = cuirve3.evaluate(t)[0, 0]
    result3[1, i] = cuirve3.evaluate(t)[1, 0]

plt.plot(result1[0, :], result1[1, :])
plt.plot(result2[0, :], result2[1, :])
plt.plot(result3[0, :], result3[1, :])
plt.show()
