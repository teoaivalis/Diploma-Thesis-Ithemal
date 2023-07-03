import torch

import numpy as np
import csv
import pandas as pd



df_hex_icl1 = pd.read_csv("/home/ithemal/ithemal/icl_uica.csv", usecols = ['hex'], low_memory = False)
hex_list_icl1 = df_hex_icl1.values.tolist()


contents = []
contents1 = []
contents2 = []
contents4 = []



with open("/home/ithemal/ithemal/throughput_icl.csv", 'r') as file2:
  csvreader = csv.reader(file2)
  count = 0
  for row in csvreader:
  	if(count % 2 == 0):
  		contents1.append(row[0])
  		count = count + 1
  	else:
    		contents2.append(row[0])
    		count = count + 1
    
  

with open("/home/ithemal/ithemal/output_icl_1.txt", 'r') as file3:
  csvreader = csv.reader(file3)
  for row in csvreader:
    contents4.append(row[0])


#content1 = file1.readlines()
#content2 = file2.readlines()
content3 = "None"
#content4 = file3.readlines()





lista = []
for i in range (len(hex_list_icl1)):
	if(float(contents2[i])==0 ):
		x = (contents1[i], float(contents2[i]), content3, contents4[i])
	else:
		x = (contents1[i], float(contents2[i]), content3, contents4[i])
		lista.append(x)



print(lista)
torch.save(lista, "tensor_icl_harness.data")
