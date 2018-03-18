import self_read_data as rd
import numpy as np
import pandas as pd

#labels = rd.get_labels()

indx = np.random.randint(1000, size=10)
labels_file  = '../data/labels.csv'
tags = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
labels = pd.read_csv(labels_file,header=0)

for i in labels['path'][indx]:
    print(i)
