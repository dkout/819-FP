import numpy as np
f = open('data_reduced.txt', 'r')
train = open('train_reduced.txt', 'w')
val = open('validation_reduced.txt', 'w')
data =[]
for i in f:
    data.append(i)
# print(data)
traincount = 0
valcount = 0

for i in data:
    if np.random.random() < 0.98:
        train.write(i)
        traincount+=1
    else:
        val.write(i)
        valcount+=1

print(traincount)
print(valcount)