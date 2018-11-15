'''
File used to visualize any data on matplotlib
'''
import matplotlib.pyplot as plt

data = [[16.48739948, 0.45256217], [-1.06479346, -2.36487147], [-1.13284854, -2.30592198],
[-1.14349095, -2.17521883], [-1.15666709, -2.10452457], [-1.16928412, -2.04967396],
[-1.67693742,  0.61209085], [-1.67247499, 0.63069504], [-1.66852032, 0.63200271]]

x = [point[0] for point in data]
y = [point[1] for point in data]

plt.scatter(x, y)
# plt.xlim(-15,15)
# plt.ylim(-15,15)
plt.show()
