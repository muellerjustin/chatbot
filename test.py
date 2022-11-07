import numpy as np
liste = [[[1, 2, 3], ["Elefant", "Hase", "Affe"]], [[4, 5, 6], ["Huhn", "Frosch", "Kamel"]]]
liste = np.array(liste)

train_x = list(liste[:,0])
train_y = list(liste[:,1])
print(train_y)