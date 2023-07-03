import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from collections import defaultdict
import csv
import pandas as pd
from scipy.spatial import distance
from numpy.linalg import norm
from statistics import median
import statistics
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#############################################################################################################################################
def euclidean_distance(x,y):
  return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
#########################################bdw model on haswell data############################################################################
print("bdw model on haswell data harness")
with open('/home/delluser/Documents/ithemal/eval_result_bdw_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
      
acc = []
help_error = []
error = []
values1 = []
values2 = []
new_values1 = []
new_values2 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value1 = list[1]
	value2 = lines2[i]
	if (value1 != "fail\n" and float(value2)!= 0.0):
		help = float(value1)/float(value2)
		values1.append(float(value1))
		values2.append(float(value2))
		acc.append(help)
		help2 = (abs(float(value1) - float(value2))) / float(value2)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values1.append(values1[count])
    	new_values2.append(values2[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values2,new_values1)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values1,new_values2)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values1,new_values2))
mse = mean_squared_error(new_values1,new_values2)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values1,new_values2)/(norm(new_values1)*norm(new_values2))
print("cosine:",cosine)
eu = euclidean_distance(new_values1,new_values2)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values1,'lst2Title': new_values2})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "HSW Values"] 
y_axis_labels = ["Move BDW model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Brodwell model to Haswell machine(Harness)")
plt.savefig("heatmap_eval_result_bdw_with_hswdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Brodwell model on Haswell data(harness)")
plt.savefig("eval_result_bdw_with_hswdata_harn.png")
#plt.show()
##############################################bdw model on skylake data########################################################################
print("bdw model on skylake data harness")
with open('/home/delluser/Documents/ithemal/eval_result_bdw_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
   
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
    
acc = []
help_error = []
error = []
values3 = []
values4 = []
new_values3 = []
new_values4 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value3 = list[1]
	value4 = lines2[i]
	if (value3 != "fail\n" and float(value4)!= 0.0):
		help = float(value3)/float(value4)
		values3.append(float(value3))
		values4.append(float(value4))
		acc.append(help)
		help2 = (abs(float(value3) - float(value4))) / float(value4)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values3.append(values3[count])
    	new_values4.append(values4[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values4,new_values3)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values3,new_values4)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values3,new_values4))
mse = mean_squared_error(new_values3,new_values4)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values3,new_values4)/(norm(new_values3)*norm(new_values4))
print("cosine:",cosine)
eu = euclidean_distance(new_values3,new_values4)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values3,'lst2Title': new_values4})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "SKL Values"] 
y_axis_labels = ["Move BDW model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Brodwell model to Skylake machine(Harness)")
plt.savefig("heatmap_eval_result_bdw_with_skldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Brodwell model on Skylake data(harness)")
plt.savefig("eval_result_bdw_with_skldata_harn.png")
#plt.show()
##############################################bdw model on icelake data########################################################################
print("bdw model on icelake data harness")
with open('/home/delluser/Documents/ithemal/eval_result_bdw_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
   
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
    
acc = []
help_error = []
error = []
values5 = []
values6 = []
new_values5 = []
new_values6 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value5 = list[1]
	value6 = lines2[i]
	if (value5 != "fail\n" and float(value6)!= 0.0):
		help = float(value5)/float(value6)
		values5.append(float(value5))
		values6.append(float(value6))
		acc.append(help)
		help2 = (abs(float(value5) - float(value6))) / float(value6)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values5.append(values5[count])
    	new_values6.append(values6[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values6,new_values5)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values5,new_values6)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values5,new_values6))
mse = mean_squared_error(new_values5,new_values6)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values5,new_values6)/(norm(new_values5)*norm(new_values6))
print("cosine:",cosine)
eu = euclidean_distance(new_values5,new_values6)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values5,'lst2Title': new_values6})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "ICL Values"] 
y_axis_labels = ["Move BDW model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Brodwell model to Icelake machine(Harness)")
plt.savefig("heatmap_eval_result_bdw_with_icldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Brodwell model on Icelake data(harness)")
plt.savefig("eval_result_bdw_with_icldata_harn.png")
#plt.show()
########################################hsw model on brodwell data#################################################################################
print("hsw model on brodwell data harness")
with open('/home/delluser/Documents/ithemal/eval_result_hsw_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
   
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
    
acc = []
help_error = []
error = []
values7 = []
values8 = []
new_values7 = []
new_values8 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value7 = list[1]
	value8 = lines2[i]
	if (value7 != "fail\n" and float(value8)!= 0.0):
		help = float(value7)/float(value8)
		values7.append(float(value7))
		values8.append(float(value8))
		acc.append(help)
		help2 = (abs(float(value7) - float(value8))) / float(value8)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values7.append(values7[count])
    	new_values8.append(values8[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values8,new_values7)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values7,new_values8)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values7,new_values8))
mse = mean_squared_error(new_values7,new_values8)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values7,new_values8)/(norm(new_values7)*norm(new_values8))
print("cosine:",cosine)
eu = euclidean_distance(new_values7,new_values8)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values7,'lst2Title': new_values8})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move HSW model", "BDW Values"] 
y_axis_labels = ["Move HSW model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Haswell model to Brodwell machine(Harness)")
plt.savefig("heatmap_eval_result_hsw_with_bdwdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Haswell model on Brodwell data(harness)")
plt.savefig("eval_result_hsw_with_bdwdata_harn.png")
#plt.show()
##################################hsw model on skylake data##########################################################################################
print("hsw model on skylake data harness")
with open('/home/delluser/Documents/ithemal/eval_result_hsw_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
   
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
    
acc = []
help_error = []
error = []
values9 = []
values10 = []
new_values9 = []
new_values10 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value9 = list[1]
	value10 = lines2[i]
	if (value9 != "fail\n" and float(value10)!= 0.0):
		help = float(value9)/float(value10)
		values9.append(float(value9))
		values10.append(float(value10))
		acc.append(help)
		help2 = (abs(float(value9) - float(value10))) / float(value10)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values9.append(values9[count])
    	new_values10.append(values10[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values10,new_values9)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values9,new_values10)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values9,new_values10))
mse = mean_squared_error(new_values9,new_values10)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values9,new_values10)/(norm(new_values9)*norm(new_values10))
print("cosine:",cosine)
eu = euclidean_distance(new_values9,new_values10)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values9,'lst2Title': new_values10})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move HSW model", "SKL Values"] 
y_axis_labels = ["Move HSW model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Haswell model to Skylake machine(Harness)")
plt.savefig("heatmap_eval_result_hsw_with_skldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Haswell model on Skylake data(harness)")
plt.savefig("eval_result_hsw_with_skldata_harn.png")
#plt.show()
##################################hsw model on icelake data##########################################################################################
print("hsw model on icelake data harness")
with open('/home/delluser/Documents/ithemal/eval_result_hsw_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
   
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
    
acc = []
help_error = []
error = []
values11 = []
values12 = []
new_values11 = []
new_values12 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value11 = list[1]
	value12 = lines2[i]
	if (value11 != "fail\n" and float(value12)!= 0.0):
		help = float(value11)/float(value12)
		values11.append(float(value11))
		values12.append(float(value12))
		acc.append(help)
		help2 = (abs(float(value11) - float(value12))) / float(value12)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values11.append(values11[count])
    	new_values12.append(values12[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values12,new_values11)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values11,new_values12)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values11,new_values12))
mse = mean_squared_error(new_values11,new_values12)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values11,new_values12)/(norm(new_values11)*norm(new_values12))
print("cosine:",cosine)
eu = euclidean_distance(new_values11,new_values12)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values11,'lst2Title': new_values12})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move HSW model", "ICL Values"] 
y_axis_labels = ["Move HSW model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Haswell model to Icelake machine(Harness)")
plt.savefig("heatmap_eval_result_hsw_with_icldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Haswell model on Icelake data(harness)")
plt.savefig("eval_result_hsw_with_icldata_harn.png")
#plt.show()
################################skl model on brodwell data###############################################################################################
print("skl model on brodwell data nano")
with open('/home/delluser/Documents/ithemal/eval_result_skl_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
      
acc = []
help_error = []
error = []
values13 = []
values14 = []
new_values13 = []
new_values14 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value13 = list[1]
	value14 = lines2[i]
	if (value13 != "fail\n" and float(value14)!= 0.0):
		help = float(value13)/float(value14)
		values13.append(float(value13))
		values14.append(float(value14))
		acc.append(help)
		help2 = (abs(float(value13) - float(value14))) / float(value14)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values13.append(values13[count])
    	new_values14.append(values14[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values14,new_values13)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values13,new_values14)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values13,new_values14))
mse = mean_squared_error(new_values13,new_values14)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values13,new_values14)/(norm(new_values13)*norm(new_values14))
print("cosine:",cosine)
eu = euclidean_distance(new_values13,new_values14)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values13,'lst2Title': new_values14})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "BDW Values"] 
y_axis_labels = ["Move SKL model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Skylake model to Brodwell machine(Nano)")
plt.savefig("heatmap_eval_result_skl_with_bdwdata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Brodwell data(Nano)")
plt.savefig("eval_result_skl_with_bdwdata_nano.png")
#plt.show()
###################################skl model on haswell data####################################################################################################
'''
with open('/home/delluser/Documents/ithemal/eval_result_skl_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values15 = []
values16 = []
new_values15 = []
new_values16 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value15 = list[1]
	value16 = lines2[i]
	if (value15 != "fail\n" and float(value16)!= 0.0):
		help = float(value15)/float(value16)
		values15.append(float(value15))
		values16.append(float(value16))
		acc.append(help)
		help2 = (abs(float(value15) - float(value16))) / float(value16)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values15.append(values15[count])
    	new_values16.append(values16[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values16,new_values15)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values15,new_values16)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values15,new_values16))
mse = mean_squared_error(new_values15,new_values16)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values15,new_values16)/(norm(new_values15)*norm(new_values16))
print("cosine:",cosine)
eu = euclidean_distance(new_values15,new_values16)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values15,'lst2Title': new_values16})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "HSW Values"] 
y_axis_labels = ["Move SKL model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Skylake model to Haswell machine")
plt.savefig("heatmap_eval_result_skl_with_hswdata.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Haswell data")
plt.savefig("eval_result_skl_with_hswdata.png")
#plt.show()
'''
###################################skl model on icelake data####################################################################################################
print("skl model on icelake data nano")
with open('/home/delluser/Documents/ithemal/eval_result_skl_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values17 = []
values18 = []
new_values17 = []
new_values18 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value17 = list[1]
	value18 = lines2[i]
	if (value17 != "fail\n" and float(value18)!= 0.0):
		help = float(value17)/float(value18)
		values17.append(float(value17))
		values18.append(float(value18))
		acc.append(help)
		help2 = (abs(float(value17) - float(value18))) / float(value18)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values17.append(values17[count])
    	new_values18.append(values18[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values18,new_values17)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values17,new_values18)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values17,new_values18))
mse = mean_squared_error(new_values17,new_values18)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values17,new_values18)/(norm(new_values17)*norm(new_values18))
print("cosine:",cosine)
eu = euclidean_distance(new_values17,new_values18)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values17,'lst2Title': new_values18})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "ICL Values"] 
y_axis_labels = ["Move SKL model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Skylake model to Icelake machine(Nano)")
plt.savefig("heatmap_eval_result_skl_with_icldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Icelake data(Nano)")
plt.savefig("eval_result_skl_with_icldata_nano.png")
#plt.show()
###################################icl model on brodwell data####################################################################################################
print("icl model on brodwell data nano")
with open('/home/delluser/Documents/ithemal/eval_result_icl_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values19 = []
values20 = []
new_values19 = []
new_values20 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value19 = list[1]
	value20 = lines2[i]
	if (value19 != "fail\n" and float(value20)!= 0.0):
		help = float(value19)/float(value20)
		values19.append(float(value19))
		values20.append(float(value20))
		acc.append(help)
		help2 = (abs(float(value19) - float(value20))) / float(value20)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values19.append(values19[count])
    	new_values20.append(values20[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values20,new_values19)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values19,new_values20)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values19,new_values20))
mse = mean_squared_error(new_values19,new_values20)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values19,new_values20)/(norm(new_values19)*norm(new_values20))
print("cosine:",cosine)
eu = euclidean_distance(new_values19,new_values20)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values19,'lst2Title': new_values20})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "BDW Values"] 
y_axis_labels = ["Move ICL model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Icelake model to Brodwell machine(Nano)")
plt.savefig("heatmap_eval_result_icl_with_bdwdata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Brodwell data(Nano)")
plt.savefig("eval_result_icl_with_bdwdata_nano.png")
#plt.show()
###################################icl model on haswell data####################################################################################################
'''
with open('/home/delluser/Documents/ithemal/eval_result_icl_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values21 = []
values22 = []
new_values21 = []
new_values22 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value21 = list[1]
	value22 = lines2[i]
	if (value21 != "fail\n" and float(value22)!= 0.0):
		help = float(value21)/float(value22)
		values21.append(float(value21))
		values22.append(float(value22))
		acc.append(help)
		help2 = (abs(float(value21) - float(value22))) / float(value22)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values21.append(values21[count])
    	new_values22.append(values22[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values22,new_values21)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values21,new_values22)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values21,new_values22))
mse = mean_squared_error(new_values21,new_values22)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values21,new_values22)/(norm(new_values21)*norm(new_values22))
print("cosine:",cosine)
eu = euclidean_distance(new_values21,new_values22)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values21,'lst2Title': new_values22})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "HSW Values"] 
y_axis_labels = ["Move ICL model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Icelake model to Haswell machine")
plt.savefig("heatmap_eval_result_icl_with_hswdata.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Haswell data")
plt.savefig("eval_result_icl_with_hswdata.png")
#plt.show()
'''
###################################icl model on skylake data####################################################################################################
print("icl model on skylake data nano")
with open('/home/delluser/Documents/ithemal/eval_result_icl_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values23 = []
values24 = []
new_values23 = []
new_values24 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value23 = list[1]
	value24 = lines2[i]
	if (value23 != "fail\n" and float(value24)!= 0.0):
		help = float(value23)/float(value24)
		values23.append(float(value23))
		values24.append(float(value24))
		acc.append(help)
		help2 = (abs(float(value23) - float(value24))) / float(value24)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values23.append(values23[count])
    	new_values24.append(values24[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values24,new_values23)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values23,new_values24)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values23,new_values24))
mse = mean_squared_error(new_values23,new_values24)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values23,new_values24)/(norm(new_values23)*norm(new_values24))
print("cosine:",cosine)
eu = euclidean_distance(new_values23,new_values24)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values23,'lst2Title': new_values24})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "SKL Values"] 
y_axis_labels = ["Move ICL model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving Icelake model to Skylake machine(Nano)")
plt.savefig("heatmap_eval_result_icl_with_skldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Skylake data(Nano)")
plt.savefig("eval_result_icl_with_skldata_nano.png")
#plt.show()
###################################amd model on brodwell data####################################################################################################
print("amd model on brodwell data nano")
with open('/home/delluser/Documents/ithemal/eval_result_amd_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values25 = []
values26 = []
new_values25 = []
new_values26 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value25 = list[1]
	value26 = lines2[i]
	if (value25 != "fail\n" and float(value26)!= 0.0):
		help = float(value25)/float(value26)
		values25.append(float(value25))
		values26.append(float(value26))
		acc.append(help)
		help2 = (abs(float(value25) - float(value26))) / float(value26)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values25.append(values25[count])
    	new_values26.append(values26[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values26,new_values25)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values25,new_values26)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values25,new_values26))
mse = mean_squared_error(new_values25,new_values26)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values25,new_values26)/(norm(new_values25)*norm(new_values26))
print("cosine:",cosine)
eu = euclidean_distance(new_values25,new_values26)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values25,'lst2Title': new_values26})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move AMD model", "BDW Values"] 
y_axis_labels = ["Move AMD model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving AMD model to Brodwell machine(Nano)")
plt.savefig("heatmap_eval_result_amd_with_bdwdata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test AMD model on Brodwell data(Nano)")
plt.savefig("eval_result_amd_with_bdwdata_nano.png")
#plt.show()
###################################amd model on haswell data####################################################################################################
'''
with open('/home/delluser/Documents/ithemal/eval_result_amd_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values27 = []
values28 = []
new_values27 = []
new_values28 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value27 = list[1]
	value28 = lines2[i]
	if (value27 != "fail\n" and float(value28)!= 0.0):
		help = float(value27)/float(value28)
		values27.append(float(value27))
		values28.append(float(value28))
		acc.append(help)
		help2 = (abs(float(value27) - float(value28))) / float(value28)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values27.append(values27[count])
    	new_values28.append(values28[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values28,new_values27)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values27,new_values28)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values27,new_values28))
mse = mean_squared_error(new_values27,new_values28)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values27,new_values28)/(norm(new_values27)*norm(new_values28))
print("cosine:",cosine)
eu = euclidean_distance(new_values27,new_values28)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values27,'lst2Title': new_values28})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move AMD model", "HSW Values"] 
y_axis_labels = ["Move AMD model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving AMD model to Haswell machine")
plt.savefig("heatmap_eval_result_amd_with_hswdata.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test AMD model on Haswell data")
plt.savefig("eval_result_amd_with_hswdata.png")
#plt.show()
'''
###################################amd model on skylake data####################################################################################################
print("amd model on skylake data nano")
with open('/home/delluser/Documents/ithemal/eval_result_amd_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values29 = []
values30 = []
new_values29 = []
new_values30 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value29 = list[1]
	value30 = lines2[i]
	if (value29 != "fail\n" and float(value30)!= 0.0):
		help = float(value29)/float(value30)
		values29.append(float(value29))
		values30.append(float(value30))
		acc.append(help)
		help2 = (abs(float(value29) - float(value30))) / float(value30)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values29.append(values29[count])
    	new_values30.append(values30[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values30,new_values29)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values29,new_values30)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values29,new_values30))
mse = mean_squared_error(new_values29,new_values30)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values29,new_values30)/(norm(new_values29)*norm(new_values30))
print("cosine:",cosine)
eu = euclidean_distance(new_values29,new_values30)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values29,'lst2Title': new_values30})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move AMD model", "SKL Values"] 
y_axis_labels = ["Move AMD model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving AMD model to Skylake machine(Nano)")
plt.savefig("heatmap_eval_result_amd_with_skldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test AMD model on Skylake data(Nano)")
plt.savefig("eval_result_amd_with_skldata_nano.png")
#plt.show()
###################################amd model on icelake data####################################################################################################
print("amd model on icelake data nano")
with open('/home/delluser/Documents/ithemal/eval_result_amd_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values31 = []
values32 = []
new_values31 = []
new_values32 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value31 = list[1]
	value32 = lines2[i]
	if (value31 != "fail\n" and float(value32)!= 0.0):
		help = float(value31)/float(value32)
		values31.append(float(value31))
		values32.append(float(value32))
		acc.append(help)
		help2 = (abs(float(value31) - float(value32))) / float(value32)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values31.append(values31[count])
    	new_values32.append(values32[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values32,new_values31)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values31,new_values32)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values31,new_values32))
mse = mean_squared_error(new_values31,new_values32)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values31,new_values32)/(norm(new_values31)*norm(new_values32))
print("cosine:",cosine)
eu = euclidean_distance(new_values31,new_values32)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values31,'lst2Title': new_values32})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move AMD model", "ICL Values"] 
y_axis_labels = ["Move AMD model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving AMD model to Icelake machine(Nano)")
plt.savefig("heatmap_eval_result_amd_with_icldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test AMD model on Icelake data(Nano)")
plt.savefig("eval_result_amd_with_icldata_nano.png")
#plt.show()
##################################################################################################################################################################
print("skl model on haswell data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_skl_harn_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values33 = []
values34 = []
new_values33 = []
new_values34 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value33 = list[1]
	value34 = lines2[i]
	if (value33 != "fail\n" and float(value34)!= 0.0):
		help = float(value33)/float(value34)
		values33.append(float(value33))
		values34.append(float(value34))
		acc.append(help)
		help2 = (abs(float(value33) - float(value34))) / float(value34)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values33.append(values33[count])
    	new_values34.append(values34[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values34,new_values33)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values33,new_values34)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values33,new_values34))
mse = mean_squared_error(new_values33,new_values34)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values33,new_values34)/(norm(new_values33)*norm(new_values34))
print("cosine:",cosine)
eu = euclidean_distance(new_values33,new_values34)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values33,'lst2Title': new_values34})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "HSW Values"] 
y_axis_labels = ["Move SKL model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving SKL model to Haswell machine(Harness)")
plt.savefig("heatmap_eval_result_skl_with_hswdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Haswell data(harness)")
plt.savefig("eval_result_skl_with_hswdata_harn.png")
#plt.show()
##################################################################################################################################################################
print("icl model on haswell data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_icl_harn_with_hswdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values35 = []
values36 = []
new_values35 = []
new_values36 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value35 = list[1]
	value36 = lines2[i]
	if (value35 != "fail\n" and float(value36)!= 0.0):
		help = float(value35)/float(value36)
		values35.append(float(value35))
		values36.append(float(value36))
		acc.append(help)
		help2 = (abs(float(value35) - float(value36))) / float(value36)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values35.append(values35[count])
    	new_values36.append(values36[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values36,new_values35)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values35,new_values36)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values35,new_values36))
mse = mean_squared_error(new_values35,new_values36)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values35,new_values36)/(norm(new_values35)*norm(new_values36))
print("cosine:",cosine)
eu = euclidean_distance(new_values35,new_values36)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values35,'lst2Title': new_values36})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "HSW Values"] 
y_axis_labels = ["Move ICL model", "HSW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving ICL model to Haswell machine(Harness)")
plt.savefig("heatmap_eval_result_icl_with_hswdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Haswell data(harness)")
plt.savefig("eval_result_icl_with_hswdata_harn.png")
#plt.show()
##################################################################################################################################################################
print("skl model on brodwell data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_skl_harn_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values37 = []
values38 = []
new_values37 = []
new_values38 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value37 = list[1]
	value38 = lines2[i]
	if (value37 != "fail\n" and float(value38)!= 0.0):
		help = float(value37)/float(value38)
		values37.append(float(value37))
		values38.append(float(value38))
		acc.append(help)
		help2 = (abs(float(value37) - float(value38))) / float(value38)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values37.append(values37[count])
    	new_values38.append(values38[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values38,new_values37)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values37,new_values38)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values37,new_values38))
mse = mean_squared_error(new_values37,new_values38)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values37,new_values38)/(norm(new_values37)*norm(new_values38))
print("cosine:",cosine)
eu = euclidean_distance(new_values37,new_values38)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values37,'lst2Title': new_values38})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "BDW Values"] 
y_axis_labels = ["Move SKL model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving SKL model to Broadwell machine(Harness)")
plt.savefig("heatmap_eval_result_skl_with_bdwdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Broadwell data(harness)")
plt.savefig("eval_result_skl_with_bdwdata_harn.png")
#plt.show()
##################################################################################################################################################################
print("icl model on brodwell data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_icl_harn_with_bdwdata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_bdw.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values39 = []
values40 = []
new_values39 = []
new_values40 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value39 = list[1]
	value40 = lines2[i]
	if (value39 != "fail\n" and float(value40)!= 0.0):
		help = float(value39)/float(value40)
		values39.append(float(value39))
		values40.append(float(value40))
		acc.append(help)
		help2 = (abs(float(value39) - float(value40))) / float(value40)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values39.append(values39[count])
    	new_values40.append(values40[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values40,new_values39)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values39,new_values40)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values39,new_values40))
mse = mean_squared_error(new_values39,new_values40)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values39,new_values40)/(norm(new_values39)*norm(new_values40))
print("cosine:",cosine)
eu = euclidean_distance(new_values39,new_values40)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values39,'lst2Title': new_values40})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "BDW Values"] 
y_axis_labels = ["Move ICL model", "BDW Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving ICL model to Broadwell machine(Harness)")
plt.savefig("heatmap_eval_result_icl_with_bdwdata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Broadwell data(harness)")
plt.savefig("eval_result_icl_with_bdwdata_harn.png")
#plt.show()
##################################################################################################################################################################
print("skl model on icelake data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_skl_harn_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values41 = []
values42 = []
new_values41 = []
new_values42 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value41 = list[1]
	value42 = lines2[i]
	if (value41 != "fail\n" and float(value42)!= 0.0):
		help = float(value41)/float(value42)
		values41.append(float(value41))
		values42.append(float(value42))
		acc.append(help)
		help2 = (abs(float(value41) - float(value42))) / float(value42)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values41.append(values41[count])
    	new_values42.append(values42[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values42,new_values41)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values41,new_values42)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values41,new_values42))
mse = mean_squared_error(new_values41,new_values42)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values41,new_values42)/(norm(new_values41)*norm(new_values42))
print("cosine:",cosine)
eu = euclidean_distance(new_values41,new_values42)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values41,'lst2Title': new_values42})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "ICL Values"] 
y_axis_labels = ["Move SKL model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving SKL model to Icelake machine(Harness)")
plt.savefig("heatmap_eval_result_skl_with_icldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on Icelake data(harness)")
plt.savefig("eval_result_skl_with_icldata_harn.png")
#plt.show()
##################################################################################################################################################################
print("bdw model on icelake data nano")
with open('/home/delluser/Documents/ithemal/eval/eval_results_bdw_nano_with_icldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_icl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values43 = []
values44 = []
new_values43 = []
new_values44 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value43 = list[1]
	value44 = lines2[i]
	if (value43 != "fail\n" and float(value44)!= 0.0):
		help = float(value43)/float(value44)
		values43.append(float(value43))
		values44.append(float(value44))
		acc.append(help)
		help2 = (abs(float(value43) - float(value44))) / float(value44)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values43.append(values43[count])
    	new_values44.append(values44[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values44,new_values43)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values43,new_values44)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values43,new_values44))
mse = mean_squared_error(new_values43,new_values44)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values43,new_values44)/(norm(new_values43)*norm(new_values44))
print("cosine:",cosine)
eu = euclidean_distance(new_values43,new_values44)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values43,'lst2Title': new_values44})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "ICL Values"] 
y_axis_labels = ["Move BDW model", "ICL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving BDW model to Icelake machine(Nano)")
plt.savefig("heatmap_eval_result_bdw_with_icldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Broadwell model on Icelake data(Nano)")
plt.savefig("eval_result_bdw_with_icldata_nano.png")
#plt.show()
##################################################################################################################################################################
print("icl model on skylake data harness")
with open('/home/delluser/Documents/ithemal/eval/eval_results_icl_harn_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values45 = []
values46 = []
new_values45 = []
new_values46 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value45 = list[1]
	value46 = lines2[i]
	if (value45 != "fail\n" and float(value46)!= 0.0):
		help = float(value45)/float(value46)
		values45.append(float(value45))
		values46.append(float(value46))
		acc.append(help)
		help2 = (abs(float(value45) - float(value46))) / float(value46)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values45.append(values45[count])
    	new_values46.append(values46[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values46,new_values45)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values45,new_values46)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values45,new_values46))
mse = mean_squared_error(new_values45,new_values46)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values45,new_values46)/(norm(new_values45)*norm(new_values46))
print("cosine:",cosine)
eu = euclidean_distance(new_values45,new_values46)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values45,'lst2Title': new_values46})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "SKL Values"] 
y_axis_labels = ["Move ICL model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving ICL model to Skylake machine(Harness)")
plt.savefig("heatmap_eval_result_icl_with_skldata_harn.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on Skylake data(harness)")
plt.savefig("eval_result_icl_with_skldata_harn.png")
#plt.show()
##################################################################################################################################################################
print("bdw model on skylake data nano")
with open('/home/delluser/Documents/ithemal/eval/eval_results_bdw_nano_with_skldata.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_skl.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values47 = []
values48 = []
new_values47 = []
new_values48 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value47 = list[1]
	value48 = lines2[i]
	if (value47 != "fail\n" and float(value48)!= 0.0):
		help = float(value47)/float(value48)
		values47.append(float(value47))
		values48.append(float(value48))
		acc.append(help)
		help2 = (abs(float(value47) - float(value48))) / float(value48)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values47.append(values47[count])
    	new_values48.append(values48[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(acc)))
from scipy import stats
res = stats.spearmanr(new_values48,new_values47)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values47,new_values48)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values47,new_values48))
mse = mean_squared_error(new_values47,new_values48)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values47,new_values48)/(norm(new_values47)*norm(new_values48))
print("cosine:",cosine)
eu = euclidean_distance(new_values47,new_values48)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values47,'lst2Title': new_values48})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "SKL Values"] 
y_axis_labels = ["Move BDW model", "SKL Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving BDW model to Skylake machine(Nano)")
plt.savefig("heatmap_eval_result_bdw_with_skldata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Broadwell model on Skylake data(Nano)")
plt.savefig("eval_result_bdw_with_skldata_nano.png")
#plt.show()
##################################################################################################################################################################
print("bdw model on amd data nano")
with open('/home/delluser/Documents/ithemal/eval/eval_results_bdw_nano_with_amd_data.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_amd.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values47 = []
values48 = []
new_values47 = []
new_values48 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value47 = list[1]
	value48 = lines2[i]
	if (value47 != "fail\n" and float(value48)!= 0.0):
		help = float(value47)/float(value48)
		values47.append(float(value47))
		values48.append(float(value48))
		acc.append(help)
		help2 = (abs(float(value47) - float(value48))) / float(value48)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values47.append(values47[count])
    	new_values48.append(values48[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_values48,new_values47)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values47,new_values48)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values47,new_values48))
mse = mean_squared_error(new_values47,new_values48)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values47,new_values48)/(norm(new_values47)*norm(new_values48))
print("cosine:",cosine)
eu = euclidean_distance(new_values47,new_values48)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values47,'lst2Title': new_values48})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move BDW model", "AMD Values"] 
y_axis_labels = ["Move BDW model", "AMD Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving BDW model to AMD machine(Nano)")
plt.savefig("heatmap_eval_result_bdw_with_amddata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Broadwell model on AMD data(Nano)")
plt.savefig("eval_result_bdw_with_amddata_nano.png")
#plt.show()
##################################################################################################################################################################
print("skl model on amd data nano")
with open('/home/delluser/Documents/ithemal/eval/eval_results_skl_nano_with_amd_data.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_amd.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values47 = []
values48 = []
new_values47 = []
new_values48 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value47 = list[1]
	value48 = lines2[i]
	if (value47 != "fail\n" and float(value48)!= 0.0):
		help = float(value47)/float(value48)
		values47.append(float(value47))
		values48.append(float(value48))
		acc.append(help)
		help2 = (abs(float(value47) - float(value48))) / float(value48)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values47.append(values47[count])
    	new_values48.append(values48[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_values48,new_values47)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values47,new_values48)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values47,new_values48))
mse = mean_squared_error(new_values47,new_values48)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values47,new_values48)/(norm(new_values47)*norm(new_values48))
print("cosine:",cosine)
eu = euclidean_distance(new_values47,new_values48)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values47,'lst2Title': new_values48})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move SKL model", "AMD Values"] 
y_axis_labels = ["Move SKL model", "AMD Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving SKL model to AMD machine(Nano)")
plt.savefig("heatmap_eval_result_bdw_with_amddata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Skylake model on AMD data(Nano)")
plt.savefig("eval_result_skl_with_amddata_nano.png")
#plt.show()
##################################################################################################################################################################
print("icl model on amd data nano")
with open('/home/delluser/Documents/ithemal/eval/eval_results_icl_nano_with_amd_data.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)

with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_amd.csv','r') as f2:
    lines2 = f2.readlines()
    x2 = len(lines2)
     
acc = []
help_error = []
error = []
values47 = []
values48 = []
new_values47 = []
new_values48 = []
q1 = 0
q3 = 0
count = 0
for i in range (1, x1+1, 2):
	list = lines1[i].split(",")
	value47 = list[1]
	value48 = lines2[i]
	if (value47 != "fail\n" and float(value48)!= 0.0):
		help = float(value47)/float(value48)
		values47.append(float(value47))
		values48.append(float(value48))
		acc.append(help)
		help2 = (abs(float(value47) - float(value48))) / float(value48)
		help_error.append(help2)

q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_values47.append(values47[count])
    	new_values48.append(values48[count])
    	error.append(help_error[count])
    	count = count + 1
median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_values48,new_values47)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_values47,new_values48)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_values47,new_values48))
mse = mean_squared_error(new_values47,new_values48)
print("mse:", mse)
print("error:", sum(error)/len(error))
cosine = np.dot(new_values47,new_values48)/(norm(new_values47)*norm(new_values48))
print("cosine:",cosine)
eu = euclidean_distance(new_values47,new_values48)
print("euclidean distance:",eu)
percentile_list = pd.DataFrame({'lst1Title': new_values47,'lst2Title': new_values48})
l2 = percentile_list
corr = l2.corr()
x_axis_labels = ["Move ICL model", "AMD Values"] 
y_axis_labels = ["Move ICL model", "AMD Values"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap for moving ICL model to AMD machine(Nano)")
plt.savefig("heatmap_eval_result_icl_with_amddata_nano.png")
#plt.show()
sns.distplot(acc, hist=True)
plt.xlim(0, 2)
plt.title("Test Icelake model on AMD data(Nano)")
plt.savefig("eval_result_icl_with_amddata_nano.png")
#plt.show()
##################################################################################################################################################################
