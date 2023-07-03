import torch
import numpy as np
import csv
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
from datasets import Dataset
from json import loads, dumps

df_hex_bdw = pd.read_csv("/home/delluser/Documents/ithemal/Ithemal/hsw_uica.csv", usecols = ['hex'], low_memory = False)
df_asm_bdw = pd.read_csv("/home/delluser/Documents/ithemal/Ithemal/hsw_uica.csv", usecols = ['asm'], low_memory = False)
hex_list_bdw1 = df_hex_bdw.values.tolist()
asm_list_bdw1 = df_asm_bdw.values.tolist()

contents1 = [] #hex
contents2 = [] #score
contents4 = [] #asm

with open("/home/delluser/Documents/ithemal/Ithemal/nanothroughput_amd.csv", 'r') as file2:
  csvreader = csv.reader(file2)
  i = 0
  for row in csvreader:
  	if (i % 2 == 1):
    		contents2.append(float(row[0]))
  	i = i + 1
  
for i in range (len(hex_list_bdw1)):
	contents1.append(hex_list_bdw1[i])
	contents4.append(asm_list_bdw1[i])
    
lista = []
for i in range (len(hex_list_bdw1)):
	l = ("id", "text", "uuid", "score")
	x = (contents1[i][0], str(contents4[i][0]), contents1[i][0], float(contents2[i]))		
	lista_dict = dict(zip(l, x))
	#json_object = json.dumps(lista_dict, indent = 4) 
	lista.append(lista_dict)

df = pd.DataFrame (lista)

train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

train.to_json(r'/home/delluser/Documents/ithemal/Ithemal/data_transformer/train_amd_nano.json',orient="records")
validate.to_json(r'/home/delluser/Documents/ithemal/Ithemal/data_transformer/validate_amd_nano.json',orient="records")
test.to_json(r'/home/delluser/Documents/ithemal/Ithemal/data_transformer/test_amd_nano.json',orient="records")
