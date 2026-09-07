import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2])
y = np.array([400, 600])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb      =  w*x[i]+b
        cost      = (f_wb-y[i])**2
        cost_sum += cost

    total_cost = (1/(2*m))*cost_sum
    return total_cost

w = 200
b = 200

f_wb = w * x + b

cost = compute_cost(x, y, w, b)
print("Cost:", cost)

'''
x	y
1	400
2	600

y=wx+b

w(1)+b=400
w(2)+b=600

Subtracting the equations:
w=200

Substituting back:
200+b=400
b=200
'''

plt.scatter(x, y, marker='x',c='red', label='Training Data')
plt.plot(x, f_wb, c='blue', label='Prediction')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

plt.show()
