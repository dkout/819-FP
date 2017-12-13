text_og = "THIS is a test. THIS is a test. Dimitris Koutentakis. Ekin Karasan. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."
letterNum = len(text_og)

Calibri1 = "THlS iS a teSt. THlS jS a teSt. DjmitriS KoutentakjS. Ekin KaraSan. The quiCk brOWn fOX jumbS OVer the laZy dOg. The quiCk broWn foX jumbS over the Iazy dog. THE QUlCK BROWN FOx JUMPS OVER THE LAZY DOG. THE QUlCK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."

TNR1 = "THIS iS a teSt. THIS iS a test. Ekin KaraSan. DimitriS KoutentakiS. The quiCk brown w-X jumpS oVer the laZy dog. The quick br0wn f0x jumpS 0ver the lazy d0g. THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THE QUICK BROWN FOX JUMPS oVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."

Candara1 = "THlS iS a teSt. THls iS a teSt. Ekin KaraSan. DimitriS KOutentakiS. The quiCk brOWn fOx jumpS Over the laZy dOg. The quiCk brown fOx jumpS over the Iazy dOg. THE QUlcK BRoWN Fox JUMPs oVER THE LAZY DoG. THE QUlCK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 O. 1 2 3 4 5 6 7 8 9 O."

Calibri2 = "THlS iS a teSt. THlS iS a teSt. DimitriS KoutentakiS. Ekin KaraSan. The quick brOWn fOx jumpS oVer the laZy dog. The quiCk broWn foX jumpS over the lazy dog. THE QUlCK BROWN FoX jUMPS OVER THE LAZY DoG. THE QUlCK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."
TNR2 = "THIS iS a teSt. THIS iS a teSt. Ekin KaraSan. DimitriS KoutentakiS. The quick brown m-X jumpS oVer the laZy dog. The quick brOwn fOX jumpS 0Ver the lazy d0g. THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THE QUICK BROwN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."
Candara2 = "THlS iS a teSt. THlS iS a teSt. Ekin KaraSan. DimitriS KoUtentakiS. The qUiCk broWn fOx jUmpS OVer the laZy dog. The quiCk brOWn fOx jumpS OVer the laZy dOg. THE QUlCK BROWN FOX JUMPS oVER THE LAZY DOG. THE QUlCK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 O. 1 2 3 4 5 6 7 8 9 O."

Calibri3 = "THIS is a test. THlS is a test. Dimitris Koutentakis. Ekin KaraSan. The quick broWn foX jumps oVer the laZy dog. The quick broWn foX jumpS over the laZy dog. THE QUlCK BROWN FOX JUvPS OVER THE LAzY DOG. THE QUlCK BROWN FOX JUvPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."
TNR3 = "THIS iS a teSt. THIS iS a teSt. Ekin KaraSan. DimitriS KoutentakiS. The quick broWn i-x jumpS over the laZy dog. The quick br0Wn f0x jumpS 0ver the lazy d0g. THE QUICK BRoWN FoX JUMPS oVER THE LAZY DoG. THE QUICK BRowN FoX JUMPS OVER THE LAZY DoG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."
Candara3 = "THlS iS a teSt. THlS iS a test. Ekin KaraSan. DimitriS KoUtentakiS. The qUiCk broWn fOX jUmpS OVer the laZy dog. The quiCk broWn fOx jumpS over the laZy dOg. THE QUlCK BROWN FOX JUMPS OVER THE LAzY DOG. THE QUlCK BRoWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 o."


Calibri4 = "THlS iS a teSt. THls is a test. Dimitris Koutentakis. Ekin KaraSan. The quick broWn fox jumpS over the laZy dog. The quick broWn foX jumps over the lazy dog. THE QUICK BROWN FOX JUMPs OVER THE LAZY DOG. THE QUlCK BROWN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."

TNR4 =  "THIS iS a teSt. THIS iS a teSt. Ekin KaraSan. DimitriS KoutentakiS. The quick brown noX jumpS over the laZy dog. The quick brOwn foX jumpS Over the lazy dOg. THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. THE QUICK BROwN FOX JUMPS OVER THE LAZY DOG. 1 2 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 0."

Candara4 = "THlS iS a teSt. THIS iS a test. Ekin KaraSan. DimitriS KoUtentdkiS. The qUiCk brown fOx jUmpS Over the laZy dog. The quiCk brOwn fOx jumpS over the laZy dOg. THE QUICK BROWN FOX JUMPS OvER THE LAZY DOG. THE QUlCK BRoWN FOX JUMPS OVER THE LAZY DOG. 1 Z 3 4 5 6 7 8 9 0. 1 2 3 4 5 6 7 8 9 O."

fonts = [Calibri1, TNR1, Candara1, Calibri2, TNR2, Candara2, Calibri3, TNR3, Candara3, Calibri4, TNR4, Candara4]
font_names  = ['Calibri1', 'TNR1', 'Candara1', 'Calibri2', 'TNR2', 'Candara2', 'Calibri3', 'TNR3', 'Candara3','Calibri4', 'TNR4', 'Candara4']

accuracies = [0,0,0,0,0,0,0,0,0,0,0,0]
accuracies_insensitive = [0,0,0,0,0,0,0,0,0,0,0,0]


for i in range(letterNum):
    letter = text_og[i]
    for j in range(len(fonts)):
        font = fonts[j]
        if font[i] == letter:
            accuracies[j] += 1
        if font[i].upper() == letter.upper():
            accuracies_insensitive[j] += 1
        
for i in range(len(fonts)):
    font = font_names[i]
    accuracy = accuracies[i]/letterNum
    accuracy_ins = accuracies_insensitive[i]/letterNum
    print("Accuracy of " + str(font) + ": " + str(accuracy*100) + "%")
    print("Case insencitive accuracy of " + str(font) + ": " + str(accuracy_ins*100) + "%")


print("Total accuracy of first network: " + str(100*(accuracies[0] + accuracies[1] + accuracies[2])/(3*letterNum)) + "%")
print("Total case insensitive accuracy of first network: " + str(100*(accuracies_insensitive[0] + accuracies_insensitive[1] + accuracies_insensitive[2])/(3*letterNum)) + "%")

print("Total accuracy of second network: " + str(100*(accuracies[3] + accuracies[4] + accuracies[5])/(3*letterNum)) + "%")
print("Total case insensitive accuracy of second network: " + str(100*(accuracies_insensitive[3] + accuracies_insensitive[4] + accuracies_insensitive[5])/(3*letterNum)) + "%")

print("Total accuracy of third network: " + str(100*(accuracies[6] + accuracies[7] + accuracies[8])/(3*letterNum)) + "%")
print("Total case insensitive accuracy of third network: " + str(100*(accuracies_insensitive[6] + accuracies_insensitive[7] + accuracies_insensitive[8])/(3*letterNum)) + "%")

print("Total accuracy of fourth network: " + str(100*(accuracies[9] + accuracies[10] + accuracies[11])/(3*letterNum)) + "%")
print("Total case insensitive accuracy of fourth network: " + str(100*(accuracies_insensitive[9] + accuracies_insensitive[10] + accuracies_insensitive[11])/(3*letterNum)) + "%")