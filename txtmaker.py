import os
f = open('data.txt','w')
print("open file")
print("start walking")
folders = os.walk('../Data/Fnt/')
print("end walking")

for folder in folders:
    category = folder[0][-2:]
    print(category)
    print(folder[0])
    for picture in folder[2]:
        # print(picture)
        f.write(folder[0] +'/' + picture + ' ' + category + "\n")
f.close()