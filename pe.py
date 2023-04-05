import numpy as np

# Initialize the base cases
P = np.zeros((16, 31))
P[1][1] = 0.5
P[1][-1] = 0.5

# Compute P_{n,k} for n >= 2
for n in range(2, 16):
    P[n][1:n+1] = 0.5 * P[n-1][0:n] + 0.5 * P[n-1][2:n+2]
    P[n][-1:-n-1:-1] = 0.5 * P[n-1][0:n] + 0.5 * P[n-1][2:n+2]

# Find the maximum value of k such that P_{15,k} > 1/11
k_max = np.max(np.where(P[15] > 1/11)[0])

print(k_max+1)