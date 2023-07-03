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
from scipy.spatial import distance
import statistics
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


#############################################################################################################################################
def euclidean_distance(x,y):
  return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_bdw_harn_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("bdw_harn_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_bdw_harn_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("bdw_harn_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_bdw_nano_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("bdw_nano_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_bdw_nano_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("bdw_nano_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_hsw_harn_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("hsw_harn_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_hsw_harn_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("hsw_harn_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_skl_harn_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("skl_harn_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_skl_harn_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("skl_harn_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_skl_nano_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("skl_nano_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_skl_nano_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("skl_nano_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_icl_harn_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("icl_harn_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_icl_harn_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("icl_harn_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_icl_nano_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("icl_nano_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_icl_nano_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("icl_nano_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_amd_nano_5.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("amd_nano_5")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
#############################################################################################################################################
with open('/home/delluser/Documents/ithemal/save_transf/save_transf_amd_nano_100.txt','r') as f1:
    lines1 = f1.readlines()
    x1 = len(lines1)
    
print("amd_nano_100")
acc = []
pred = []
actual = []
new_pred = []
new_actual = []
q1 = 0
q3 = 0
count = 0
error = []
for  i in range (x1):
	if (i % 2 == 0):
		pred.append(float(lines1[i]))
	else:
		actual.append(float(lines1[i]))
	


for i in range(len(pred)):
	if (float(actual[i]) == 0.0):
		help = 0.0
		help2 = 0.0
	else:
		help = float(pred[i])/float(actual[i])
		help2 = abs((pred[i] - actual[i]))/ actual[i]
	acc.append(help)
	error.append(help2)
		
q1, q3 = np.percentile(acc, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in acc:
    if ele < lower_fence or ele > higher_fence :
    	acc.remove(ele)
    	count = count + 1
    else :
    	new_pred.append(pred[count])
    	new_actual.append(actual[count])
    	count = count + 1

median_value = median(acc)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in acc])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(acc)/len(acc))
from scipy import stats
res = stats.spearmanr(new_actual,new_pred)
print("spearsman", res)
tau, p_value = stats.kendalltau(new_pred,new_actual)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_pred,new_actual))
mse = mean_squared_error(new_pred,new_actual)
print("mse:", mse)
print("error:", sum(error)/len(error))
