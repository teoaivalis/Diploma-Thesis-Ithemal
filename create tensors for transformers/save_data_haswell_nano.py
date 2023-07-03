import torch
#import torchvision
import numpy as np
import csv
import pandas as pd
import pickle



df_hex_hsw = pd.read_csv("/home/ithemal/ithemal/hsw_uica.csv", usecols = ['hex'], low_memory = False)
hex_list_hsw1 = df_hex_hsw.values.tolist()


contents = []
contents1 = []
contents2 = []
contents4 = []





with open("/home/ithemal/ithemal/throughput_hsw.csv", 'r') as file2:
  csvreader = csv.reader(file2)
  count = 0
  for row in csvreader:
  	if(count % 2 == 0):
  		contents1.append(row[0])
  		count = count + 1
  	else:
    		contents2.append(row[0])
    		count = count + 1
  
with open("/home/ithemal/ithemal/throughput_values_haswell.csv",'a') as f:
	for i in range(len(hex_list_hsw1)):
		f.write(contents2[i])
		f.write("\n")
	
	
with open("/home/ithemal/ithemal/output_hsw_1.csv", 'r') as file3:
  csvreader = csv.reader(file3)
  for row in csvreader:
    contents4.append(row[0])


content3 = "None"


lista = []
for i in range (len(hex_list_hsw1)):
	x = (contents1[i], float(contents2[i]), content3, contents4[i])
	lista.append(x)



print(lista)
torch.save(lista, "tensor_haswell.data")
