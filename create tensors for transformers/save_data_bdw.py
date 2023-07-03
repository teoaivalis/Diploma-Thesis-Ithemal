import torch
#import torchvision
import numpy as np
import csv
import pandas as pd
import pickle



#read_file = pd.read_csv (r'/home/delluser/Desktop/hexcodes_hsw_1.txt')
#read_file.to_csv (r'/home/delluser/Desktop/hexcodes_hsw_1.csv', index=None)
############read_file = pd.read_csv (r'/home/delluser/Desktop/throughput2.txt')
############read_file.to_csv (r'/home/delluser/Desktop/throughput2.csv', index=None)
#read_file = pd.read_csv (r'/home/delluser/Desktop/output_hsw_1.txt')
#read_file.to_csv (r'/home/delluser/Desktop/output_hsw_1.csv', index=None)


#file1 = open("/home/delluser/Documents/ithemal/Ithemal/haswell_sample.data", "r")
#file2 = open("/home/delluser/Desktop/throughput1.csv", "r")
#file3 = open("/home/delluser/Desktop/output_hsw_1.csv", "r")

#df_hex_bdw = pd.read_csv("/home/delluser/Documents/ithemal/uiCA-eval/bdw/bdw_uica.csv", usecols = ['hex'], low_memory = False)
df_hex_bdw = pd.read_csv("/home/ithemal/ithemal/bdw_uica.csv", usecols = ['hex'], low_memory = False)
hex_list_bdw1 = df_hex_bdw.values.tolist()


contents = []
contents1 = []
contents2 = []
contents4 = []

#with open("/home/delluser/Desktop/hexcodes_hsw_1.csv", 'r') as file1:
#  csvreader = csv.reader(file1)
#  for row in csvreader:
#    contents1.append(row)






#with open("/home/delluser/Desktop/throughput2.csv", 'r') as file2:
with open("/home/ithemal/ithemal/throughput2.csv", 'r') as file2:
  csvreader = csv.reader(file2)
  for row in csvreader:
    contents2.append(float(row[0]))
    
  
#with open("/home/delluser/Documents/ithemal/Ithemal/data_collection/build/bin/output_bdw_1.csv", 'r') as file3:
with open("/home/ithemal/ithemal/output_bdw_1.csv", 'r') as file3:
  csvreader = csv.reader(file3)
  for row in csvreader:
    contents4.append(row[0])


#content1 = file1.readlines()
#content2 = file2.readlines()
content3 = "None"
#content4 = file3.readlines()


#for i in content1:
#    contents1.append(i.rstrip('\n'))
#print(contents1)



#for i in content2:
#    contents2.append(i.rstrip('\n'))


#for i in content4:
#    contents4.append(i.rstrip('\n'))




lista = []
for i in range (len(hex_list_bdw1)):
	contents1.append(hex_list_bdw1[i])
	x = (contents1[i][0], contents2[i], content3, contents4[i])
	lista.append(x)

#file = open('important.data', 'wb')

print(lista)
torch.save(lista, "tensor_bdw.data")
#pickle.dump(lista, file)
