import numpy as np

x = np.random.random((3,5,4))
y = np.random.random((1,5,4))
z = np.random.random((1,5,4))
l = [x, y, z]

m = x
print(m.shape)
for i in range(1, len(l)):
    m = np.vstack((m, l[i]))
    print(m.shape)


print("MODIFY")