from collections import defaultdict
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from scipy.spatial import distance
from numpy.linalg import norm
import math 
from statistics import median
import statistics
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



def euclidean_distance(x,y):
  return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

df = pd.read_csv("merge_values.csv")
new_df = df.T
'''
corr = new_df.corr()
x_axis_labels = ["BDW", "HSW", "SKL", "ICL", "BDW(Nano)", "SKL(Nano)", "ICL(Nano)", "AMD(Nano)"] 
y_axis_labels = ["BDW", "HSW", "SKL", "ICL", "BDW(Nano)", "SKL(Nano)", "ICL(Nano)", "AMD(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap of real values over all processors")
plt.savefig("heatmap_of_merged_real_values.png")
plt.show()
'''
##############################################bdw_to_hsw####################################################################
bdw_to_hsw = []
help_df0 = []
help_df1 = []
new_df0 = []
new_df1 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[0][i] != 0 and new_df[1][i] != 0):
		help = float(new_df[0][i])/float(new_df[1][i])
		bdw_to_hsw.append(help)
		help_df0.append(new_df[0][i])
		help_df1.append(new_df[1][i])
q1, q3 = np.percentile(bdw_to_hsw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in bdw_to_hsw:
    if  ele == float(0) or ele < lower_fence or ele > higher_fence :
    	bdw_to_hsw.remove(ele)
    	count = count + 1
    else :
    	new_df0.append(help_df0[count])
    	new_df1.append(help_df1[count])
    	#new_df0.append(new_df[0][count])
    	#new_df1.append(new_df[1][count])
    	count =count + 1

print("bdw_to_hsw")
print(len(bdw_to_hsw))
median_value = median(bdw_to_hsw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in bdw_to_hsw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(bdw_to_hsw)))
#cosine = np.dot(new_df0,new_df1)/(norm(new_df0)*norm(new_df1))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df0,new_df1)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df0,new_df1)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df0,new_df1))
mse = mean_squared_error(new_df0,new_df1)
print("mse:", mse)
sns.distplot(bdw_to_hsw, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with haswell values")
plt.savefig("bdw_to_hsw.png")
#plt.show()
############################################bdw_to_skl###########################################################################
bdw_to_skl = []
help_df0 = []
help_df2 = []
new_df0 = []
new_df2 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[0][i] != 0 and new_df[2][i] != 0):
		help = float(new_df[0][i])/float(new_df[2][i])
		bdw_to_skl.append(help)
		help_df0.append(new_df[0][i])
		help_df2.append(new_df[2][i])
		
q1, q3 = np.percentile(bdw_to_skl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in bdw_to_skl:
    if ele < lower_fence or ele > higher_fence :
    	bdw_to_skl.remove(ele)
    	count = count + 1
    else :
    	new_df0.append(help_df0[count])
    	new_df2.append(help_df2[count])
    	#new_df0.append(new_df[0][count])
    	#new_df2.append(new_df[2][count])
    	count =count + 1
print("bdw_to_skl")
print(len(bdw_to_skl))
median_value = median(bdw_to_skl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in bdw_to_skl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(bdw_to_skl)))
#cosine = np.dot(new_df0,new_df2)/(norm(new_df0)*norm(new_df2))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df0,new_df2)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df0,new_df2)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df0,new_df2))
mse = mean_squared_error(new_df0,new_df2)
print("mse:", mse)
sns.distplot(bdw_to_skl, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with skylake values")
plt.savefig("bdw_to_skl.png")
#plt.show()
#######################################bdw_to_icl###############################################################################
bdw_to_icl = []
help_df0 = []
help_df3 = []
new_df0 = []
new_df3 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[0][i] != 0 and new_df[3][i] != 0):
		help = float(new_df[0][i])/float(new_df[3][i])
		bdw_to_icl.append(help)
		help_df0.append(new_df[0][i])
		help_df3.append(new_df[3][i])
		
q1, q3 = np.percentile(bdw_to_icl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in bdw_to_icl:
    if ele < lower_fence or ele > higher_fence :
    	bdw_to_icl.remove(ele)
    	count = count + 1
    else :
    	new_df0.append(help_df0[count])
    	new_df3.append(help_df3[count])
    	#new_df0.append(new_df[0][count])
    	#new_df3.append(new_df[3][count])
    	count =count + 1
print("bdw_to_icl")
print(len(bdw_to_icl))
median_value = median(bdw_to_icl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in bdw_to_icl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(bdw_to_icl)))
#cosine = np.dot(new_df0,new_df3)/(norm(new_df0)*norm(new_df3))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df0,new_df3)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df0,new_df3)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df0,new_df3))
mse = mean_squared_error(new_df0,new_df3)
print("mse:", mse)
sns.distplot(bdw_to_icl, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with icelake values")
plt.savefig("bdw_to_icl.png")
#plt.show()
#########################################hsw_to_bdw########################################################################################
hsw_to_bdw = []
help_df1 = []
help_df0 = []
new_df1 = []
new_df0 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[1][i] != 0 and new_df[0][i] != 0):
		help = float(new_df[1][i])/float(new_df[0][i])
		hsw_to_bdw.append(help)
		help_df1.append(new_df[1][i])
		help_df0.append(new_df[0][i])
		
q1, q3 = np.percentile(hsw_to_bdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in hsw_to_bdw:
    if ele < lower_fence or ele > higher_fence :
    	hsw_to_bdw.remove(ele)
    	count = count + 1
    else :
    	new_df1.append(help_df1[count])
    	new_df0.append(help_df0[count])
    	#new_df1.append(new_df[1][count])
    	#new_df0.append(new_df[0][count])
    	count =count + 1
print("hsw_to_bdw")
print(len(hsw_to_bdw))
median_value = median(hsw_to_bdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in hsw_to_bdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(hsw_to_bdw)))
#cosine = np.dot(new_df1,new_df0)/(norm(new_df1)*norm(new_df0))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df1,new_df0)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df1,new_df0)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df1,new_df0))
mse = mean_squared_error(new_df1,new_df0)
print("mse:", mse)
sns.distplot(hsw_to_bdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare haswell with brodwell values")
plt.savefig("hsw_to_bdw.png")
#plt.show()
#########################################hsw_to_skl########################################################################################
hsw_to_skl = []
help_df1 = []
help_df2 = []
new_df1 = []
new_df2 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[1][i] != 0 and new_df[2][i] != 0):
		help = float(new_df[1][i])/float(new_df[2][i])
		hsw_to_skl.append(help)
		help_df1.append(new_df[1][i])
		help_df2.append(new_df[2][i])
		
q1, q3 = np.percentile(hsw_to_skl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in hsw_to_skl:
    if ele < lower_fence or ele > higher_fence :
    	hsw_to_skl.remove(ele)
    	count = count + 1
    else :
    	new_df1.append(help_df1[count])
    	new_df2.append(help_df2[count])
    	#new_df1.append(new_df[1][count])
    	#new_df2.append(new_df[2][count])
    	count =count + 1
print("hsw_to_skl")
print(len(hsw_to_skl))
median_value = median(hsw_to_skl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in hsw_to_skl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(hsw_to_skl)))
#cosine = np.dot(new_df1,new_df2)/(norm(new_df1)*norm(new_df2))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df1,new_df2)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df1,new_df2)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df1,new_df2))
mse = mean_squared_error(new_df1,new_df2)
print("mse:", mse)
sns.distplot(hsw_to_skl, hist=True)
plt.xlim(0, 2)
plt.title("Compare haswell with skylake values")
plt.savefig("hsw_to_skl.png")
#plt.show()
#########################################hsw_to_icl########################################################################################
hsw_to_icl = []
help_df1 = []
help_df3 = []
new_df1 = []
new_df3 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[1][i] != 0 and new_df[3][i] != 0):
		help = float(new_df[1][i])/float(new_df[3][i])
		hsw_to_icl.append(help)
		help_df1.append(new_df[1][i])
		help_df3.append(new_df[3][i])
		
q1, q3 = np.percentile(hsw_to_icl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in hsw_to_icl:
    if ele < lower_fence or ele > higher_fence :
    	hsw_to_icl.remove(ele)
    	count = count + 1
    else :
    	new_df1.append(help_df1[count])
    	new_df3.append(help_df3[count])
    	#new_df1.append(new_df[1][count])
    	#new_df3.append(new_df[3][count])
    	count =count + 1
print("hsw_to_icl")
print(len(hsw_to_icl))
median_value = median(hsw_to_icl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in hsw_to_icl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(hsw_to_icl)))
#cosine = np.dot(new_df1,new_df3)/(norm(new_df1)*norm(new_df3))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df1,new_df3)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df1,new_df3)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df1,new_df3))
mse = mean_squared_error(new_df1,new_df3)
print("mse:", mse)
sns.distplot(hsw_to_icl, hist=True)
plt.xlim(0, 2)
plt.title("Compare haswell with icelake values")
plt.savefig("hsw_to_icl.png")
#plt.show()
#########################################skl_to_bdw########################################################################################
skl_to_bdw = []
help_df2 = []
help_df0 = []
new_df2 = []
new_df0 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[2][i] != 0 and new_df[0][i] != 0):
		help = float(new_df[2][i])/float(new_df[0][i])
		skl_to_bdw.append(help)
		help_df2.append(new_df[2][i])
		help_df0.append(new_df[0][i])
		
q1, q3 = np.percentile(skl_to_bdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in skl_to_bdw:
    if ele < lower_fence or ele > higher_fence :
    	skl_to_bdw.remove(ele)
    	count = count + 1
    else :
    	new_df2.append(help_df2[count])
    	new_df0.append(help_df0[count])
    	#new_df2.append(new_df[2][count])
    	#new_df0.append(new_df[0][count])
    	count =count + 1
print("skl_to_bdw")
print(len(skl_to_bdw))
median_value = median(skl_to_bdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in skl_to_bdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(skl_to_bdw)))
#cosine = np.dot(new_df2,new_df0)/(norm(new_df2)*norm(new_df0))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df2,new_df0)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df2,new_df0)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df2,new_df0))
mse = mean_squared_error(new_df2,new_df0)
print("mse:", mse)
sns.distplot(skl_to_bdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with brodwell values")
plt.savefig("skl_to_bdw.png")
#plt.show()
#########################################skl_to_hsw########################################################################################
skl_to_hsw = []
help_df2 = []
help_df1 = []
new_df2 = []
new_df1 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[2][i] != 0 and new_df[1][i] != 0):
		help = float(new_df[2][i])/float(new_df[1][i])
		skl_to_hsw.append(help)
		help_df2.append(new_df[2][i])
		help_df1.append(new_df[1][i])
		
q1, q3 = np.percentile(skl_to_hsw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in skl_to_hsw:
    if ele < lower_fence or ele > higher_fence :
    	skl_to_hsw.remove(ele)
    	count = count + 1
    else :
    	new_df2.append(help_df2[count])
    	new_df1.append(help_df1[count])
    	#new_df2.append(new_df[2][count])
    	#new_df1.append(new_df[1][count])
    	count =count + 1
print("skl_to_hsw")
print(len(skl_to_hsw))
median_value = median(skl_to_hsw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in skl_to_hsw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(skl_to_hsw)))
#cosine = np.dot(new_df2,new_df1)/(norm(new_df2)*norm(new_df1))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df2,new_df1)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df2,new_df1)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df2,new_df1))
mse = mean_squared_error(new_df2,new_df1)
print("mse:", mse)
sns.distplot(skl_to_hsw, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with haswell values")
plt.savefig("skl_to_hsw.png")
#plt.show()
#########################################skl_to_icl########################################################################################
skl_to_icl = []
help_df2 = []
help_df3 = []
new_df2 = []
new_df3 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[2][i] != 0 and new_df[3][i] != 0):
		help = float(new_df[2][i])/float(new_df[3][i])
		skl_to_icl.append(help)
		help_df2.append(new_df[2][i])
		help_df3.append(new_df[3][i])
		
q1, q3 = np.percentile(skl_to_icl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in skl_to_icl:
    if ele < lower_fence or ele > higher_fence :
    	skl_to_icl.remove(ele)
    	count = count + 1
    else :
    	new_df2.append(help_df2[count])
    	new_df3.append(help_df3[count])
    	#new_df2.append(new_df[2][count])
    	#new_df3.append(new_df[3][count])
    	count =count + 1
print("skl_to_icl")
print(len(skl_to_icl))
median_value = median(skl_to_icl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in skl_to_icl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(skl_to_icl)))
#cosine = np.dot(new_df2,new_df3)/(norm(new_df2)*norm(new_df3))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df2,new_df3)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df2,new_df3)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df2,new_df3))
mse = mean_squared_error(new_df2,new_df3)
print("mse:", mse)
sns.distplot(skl_to_icl, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with icelake values")
plt.savefig("skl_to_icl.png")
#plt.show()
#########################################icl_to_bdw########################################################################################
icl_to_bdw = []
help_df3 = []
help_df0 = []
new_df3 = []
new_df0 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[3][i] != 0 and new_df[0][i] != 0):
		help = float(new_df[3][i])/float(new_df[0][i])
		icl_to_bdw.append(help)
		help_df3.append(new_df[3][i])
		help_df0.append(new_df[0][i])
		
q1, q3 = np.percentile(icl_to_bdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in icl_to_bdw:
    if ele < lower_fence or ele > higher_fence :
    	icl_to_bdw.remove(ele)
    	count = count + 1
    else :
    	new_df3.append(help_df3[count])
    	new_df0.append(help_df0[count])
    	#new_df3.append(new_df[3][count])
    	#new_df0.append(new_df[0][count])
    	count =count + 1
print("icl_to_bdw")
print(len(icl_to_bdw))
median_value = median(icl_to_bdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in icl_to_bdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(icl_to_bdw)))
#cosine = np.dot(new_df3,new_df0)/(norm(new_df3)*norm(new_df0))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df3,new_df0)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df3,new_df0)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df3,new_df0))
mse = mean_squared_error(new_df3,new_df0)
print("mse:", mse)
sns.distplot(icl_to_bdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with brodwell values")
plt.savefig("icl_to_bdw.png")
#plt.show()
#########################################icl_to_hsw########################################################################################
icl_to_hsw = []
help_df3 = []
help_df1 = []
new_df3 = []
new_df1 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[3][i] != 0 and new_df[1][i] != 0):
		help = float(new_df[3][i])/float(new_df[1][i])
		icl_to_hsw.append(help)
		help_df3.append(new_df[3][i])
		help_df1.append(new_df[1][i])
		
q1, q3 = np.percentile(icl_to_hsw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in icl_to_hsw:
    if ele < lower_fence or ele > higher_fence :
    	icl_to_hsw.remove(ele)
    	count = count + 1
    else :
    	new_df3.append(help_df3[count])
    	new_df1.append(help_df1[count])
    	#new_df3.append(new_df[3][count])
    	#new_df1.append(new_df[1][count])
    	count =count + 1
print("icl_to_hsw")
print(len(icl_to_hsw))
median_value = median(icl_to_hsw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in icl_to_hsw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(icl_to_hsw)))
#cosine = np.dot(new_df3,new_df1)/(norm(new_df3)*norm(new_df1))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df3,new_df1)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df3,new_df1)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df3,new_df1))
mse = mean_squared_error(new_df3,new_df1)
print("mse:", mse)
sns.distplot(icl_to_hsw, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with haswell values")
plt.savefig("icl_to_hsw.png")
#plt.show()
#########################################icl_to_skl########################################################################################
icl_to_skl = []
help_df3 = []
help_df2 = []
new_df3 = []
new_df2 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[3][i] != 0 and new_df[2][i] != 0):
		help = float(new_df[3][i])/float(new_df[2][i])
		icl_to_skl.append(help)
		help_df3.append(new_df[3][i])
		help_df2.append(new_df[2][i])
		
q1, q3 = np.percentile(icl_to_skl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in icl_to_skl:
    if ele < lower_fence or ele > higher_fence :
    	icl_to_skl.remove(ele)
    	count = count + 1
    else :
    	new_df3.append(help_df3[count])
    	new_df2.append(help_df2[count])
    	#new_df3.append(new_df[3][count])
    	#new_df2.append(new_df[2][count])
    	count =count + 1
print("icl_to_skl")
print(len(icl_to_skl))
median_value = median(icl_to_skl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in icl_to_skl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(icl_to_skl)))
#cosine = np.dot(new_df3,new_df2)/(norm(new_df3)*norm(new_df2))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df3,new_df2)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df3,new_df2)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df3,new_df2))
mse = mean_squared_error(new_df3,new_df2)
print("mse:", mse)
sns.distplot(icl_to_skl, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with skylake values")
plt.savefig("icl_to_skl.png")
#plt.show()
#########################################nanobdw_to_nanoskl########################################################################################
nanobdw_to_nanoskl = []
help_df4 = []
help_df5 = []
new_df4 = []
new_df5 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[4][i] != 0 and new_df[5][i] != 0):
		help = float(new_df[4][i])/float(new_df[5][i])
		nanobdw_to_nanoskl.append(help)
		help_df4.append(new_df[4][i])
		help_df5.append(new_df[5][i])
		
q1, q3 = np.percentile(nanobdw_to_nanoskl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanobdw_to_nanoskl:
    if ele < lower_fence or ele > higher_fence :
    	nanobdw_to_nanoskl.remove(ele)
    	count = count + 1
    else :
    	new_df4.append(help_df4[count])
    	new_df5.append(help_df5[count])
    	#new_df4.append(new_df[4][count])
    	#new_df5.append(new_df[5][count])
    	count =count + 1
print("nanobdw_to_nanoskl")
print(len(nanobdw_to_nanoskl))
median_value = median(nanobdw_to_nanoskl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanobdw_to_nanoskl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanobdw_to_nanoskl)))
#cosine = np.dot(new_df4,new_df5)/(norm(new_df4)*norm(new_df5))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df4,new_df5)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df4,new_df5)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df4,new_df5))
mse = mean_squared_error(new_df4,new_df5)
print("mse:", mse)
sns.distplot(nanobdw_to_nanoskl, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with skylake values (nanobench)")
plt.savefig("nanobdw_to_nanoskl.png")
#plt.show()
#########################################nanobdw_to_nanoicl########################################################################################
nanobdw_to_nanoicl = []
help_df4 = []
help_df6 = []
new_df4 = []
new_df6 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[4][i] != 0 and new_df[6][i] != 0):
		help = float(new_df[4][i])/float(new_df[6][i])
		nanobdw_to_nanoicl.append(help)
		help_df4.append(new_df[4][i])
		help_df6.append(new_df[6][i])
		
q1, q3 = np.percentile(nanobdw_to_nanoicl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanobdw_to_nanoicl:
    if ele < lower_fence or ele > higher_fence :
    	nanobdw_to_nanoicl.remove(ele)
    	count = count + 1
    else :
    	new_df4.append(help_df4[count])
    	new_df6.append(help_df6[count])
    	#new_df4.append(new_df[4][count])
    	#new_df6.append(new_df[6][count])
    	count =count + 1
print("nanobdw_to_nanoicl")
print(len(nanobdw_to_nanoicl))
median_value = median(nanobdw_to_nanoicl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanobdw_to_nanoicl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanobdw_to_nanoicl)))
#cosine = np.dot(new_df4,new_df6)/(norm(new_df4)*norm(new_df6))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df4,new_df6)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df4,new_df6)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df4,new_df6))
mse = mean_squared_error(new_df4,new_df6)
print("mse:", mse)
sns.distplot(nanobdw_to_nanoicl, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with icelake values (nanobench)")
plt.savefig("nanobdw_to_nanoicl.png")
#plt.show()
#########################################nanobdw_to_nanoamd########################################################################################
nanobdw_to_nanoamd = []
help_df4 = []
help_df7 = []
new_df4 = []
new_df7 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[4][i] != 0 and float(new_df[7][i]) != 0.0):
		help = float(new_df[4][i])/float(new_df[7][i])
		nanobdw_to_nanoamd.append(help)
		help_df4.append(new_df[4][i])
		help_df7.append(new_df[7][i])
		
q1, q3 = np.percentile(nanobdw_to_nanoamd, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanobdw_to_nanoamd:
    if ele < lower_fence or ele > higher_fence :
    	nanobdw_to_nanoamd.remove(ele)
    	count = count + 1
    else :
    	new_df4.append(help_df4[count])
    	new_df7.append(help_df7[count])
    	#new_df4.append(new_df[4][count])
    	#new_df7.append(new_df[7][count])
    	count =count + 1
print("nanobdw_to_nanoamd")
print(len(nanobdw_to_nanoamd))
median_value = median(nanobdw_to_nanoamd)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanobdw_to_nanoamd])
print("median absolute",median_absolute_deviation)
print(len(nanobdw_to_nanoamd))
#print("geometrical mean ", str(statistics.geometric_mean(nanobdw_to_nanoamd)))
print("geometrical mean ", sum(nanobdw_to_nanoamd)/len(nanobdw_to_nanoamd))
#cosine = np.dot(new_df4,new_df7)/(norm(new_df4)*norm(new_df7))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df4,new_df7)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df4,new_df7)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df4,new_df7))
mse = mean_squared_error(new_df4,new_df7)
print("mse:", mse)
sns.distplot(nanobdw_to_nanoamd, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell with AMD values (nanobench)")
plt.savefig("nanobdw_to_nanoamd.png")
#plt.show()
#########################################nanoskl_to_nanobdw########################################################################################
nanoskl_to_nanobdw = []
help_df5 = []
help_df4 = []
new_df5 = []
new_df4 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[5][i] != 0 and new_df[4][i] != 0):
		help = float(new_df[5][i])/float(new_df[4][i])
		nanoskl_to_nanobdw.append(help)
		help_df5.append(new_df[5][i])
		help_df4.append(new_df[4][i])
		
q1, q3 = np.percentile(nanoskl_to_nanobdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoskl_to_nanobdw:
    if ele < lower_fence or ele > higher_fence :
    	nanoskl_to_nanobdw.remove(ele)
    	count = count + 1
    else :
    	new_df5.append(help_df5[count])
    	new_df4.append(help_df4[count])
    	#new_df5.append(new_df[5][count])
    	#new_df4.append(new_df[4][count])
    	count =count + 1
print("nanoskl_to_nanobdw")
print(len(nanoskl_to_nanobdw))
median_value = median(nanoskl_to_nanobdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoskl_to_nanobdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanoskl_to_nanobdw)))
print("geometrical mean ", sum(nanoskl_to_nanobdw)/len(nanoskl_to_nanobdw))
#cosine = np.dot(new_df5,new_df4)/(norm(new_df5)*norm(new_df4))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df5,new_df4)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df5,new_df4)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df5,new_df4))
mse = mean_squared_error(new_df5,new_df4)
print("mse:", mse)
sns.distplot(nanoskl_to_nanobdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with brodwell values (nanobench)")
plt.savefig("nanoskl_to_nanobdw.png")
#plt.show()
#########################################nanoskl_to_nanoicl########################################################################################
nanoskl_to_nanoicl = []
help_df5 = []
help_df6 = []
new_df5 = []
new_df6 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[5][i] != 0 and new_df[6][i] != 0):
		help = float(new_df[5][i])/float(new_df[6][i])
		nanoskl_to_nanoicl.append(help)
		help_df5.append(new_df[5][i])
		help_df6.append(new_df[6][i])
		
q1, q3 = np.percentile(nanoskl_to_nanoicl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoskl_to_nanoicl:
    if ele < lower_fence or ele > higher_fence :
    	nanoskl_to_nanoicl.remove(ele)
    	count = count + 1
    else :
    	new_df5.append(help_df5[count])
    	new_df6.append(help_df6[count])
    	#new_df5.append(new_df[5][count])
    	#new_df6.append(new_df[6][count])
    	count =count + 1
print("nanoskl_to_nanoicl")
print(len(nanoskl_to_nanoicl))
median_value = median(nanoskl_to_nanoicl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoskl_to_nanoicl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanoskl_to_nanoicl)))
print("geometrical mean ", sum(nanoskl_to_nanoicl)/len(nanoskl_to_nanoicl))
#cosine = np.dot(new_df5,new_df6)/(norm(new_df5)*norm(new_df6))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df5,new_df6)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df5,new_df6)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df5,new_df6))
mse = mean_squared_error(new_df5,new_df6)
print("mse:", mse)
sns.distplot(nanoskl_to_nanoicl, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with icelake values (nanobench)")
plt.savefig("nanoskl_to_nanoicl.png")
#plt.show()
#########################################nanoskl_to_nanoamd########################################################################################
nanoskl_to_nanoamd = []
help_df5 = []
help_df7 = []
new_df5 = []
new_df7 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[5][i] != 0 and new_df[7][i] != 0):
		help = float(new_df[5][i])/float(new_df[7][i])
		nanoskl_to_nanoamd.append(help)
		help_df5.append(new_df[5][i])
		help_df7.append(new_df[7][i])
		
q1, q3 = np.percentile(nanoskl_to_nanoamd, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoskl_to_nanoamd:
    if ele < lower_fence or ele > higher_fence :
    	nanoskl_to_nanoamd.remove(ele)
    	count = count + 1
    else :
    	new_df5.append(help_df5[count])
    	new_df7.append(help_df7[count])
    	#new_df5.append(new_df[5][count])
    	#new_df7.append(new_df[7][count])
    	count =count + 1
print("nanoskl_to_nanoamd")
print(len(nanoskl_to_nanoamd))
median_value = median(nanoskl_to_nanoamd)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoskl_to_nanoamd])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(nanoskl_to_nanoamd)/len(nanoskl_to_nanoamd))
#print("geometrical mean ", str(statistics.geometric_mean(nanoskl_to_nanoamd)))
#cosine = np.dot(new_df5,new_df7)/(norm(new_df5)*norm(new_df7))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df5,new_df7)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df5,new_df7)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df5,new_df7))
mse = mean_squared_error(new_df5,new_df7)
print("mse:", mse)
sns.distplot(nanoskl_to_nanoamd, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake with AMD values (nanobench)")
plt.savefig("nanoskl_to_nanoamd.png")
#plt.show()
#########################################nanoicl_to_nanobdw########################################################################################
nanoicl_to_nanobdw = []
help_df6 = []
help_df4 = []
new_df6 = []
new_df4 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[6][i] != 0 and new_df[4][i] != 0):
		help = float(new_df[6][i])/float(new_df[4][i])
		nanoicl_to_nanobdw.append(help)
		help_df6.append(new_df[6][i])
		help_df4.append(new_df[4][i])
		
q1, q3 = np.percentile(nanoicl_to_nanobdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoicl_to_nanobdw:
    if ele < lower_fence or ele > higher_fence :
    	nanoicl_to_nanobdw.remove(ele)
    	count = count + 1
    else :
    	new_df6.append(help_df6[count])
    	new_df4.append(help_df4[count])
    	#new_df6.append(new_df[6][count])
    	#new_df4.append(new_df[4][count])
    	count =count + 1
print("nanoicl_to_nanobdw")
print(len(nanoicl_to_nanobdw))
median_value = median(nanoicl_to_nanobdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoicl_to_nanobdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanoicl_to_nanobdw)))
print("geometrical mean ", sum(nanoicl_to_nanobdw)/len(nanoicl_to_nanobdw))
#cosine = np.dot(new_df6,new_df4)/(norm(new_df6)*norm(new_df4))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df6,new_df4)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df6,new_df4)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df6,new_df4))
mse = mean_squared_error(new_df6,new_df4)
print("mse:", mse)
sns.distplot(nanoicl_to_nanobdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with brodwell values (nanobench)")
plt.savefig("nanoicl_to_nanobdw.png")
#plt.show()
#########################################nanoicl_to_nanoskl########################################################################################
nanoicl_to_nanoskl = []
help_df6 = []
help_df5 = []
new_df6 = []
new_df5 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[6][i] != 0 and new_df[5][i] != 0):
		help = float(new_df[6][i])/float(new_df[5][i])
		nanoicl_to_nanoskl.append(help)
		help_df6.append(new_df[6][i])
		help_df5.append(new_df[5][i])
		
q1, q3 = np.percentile(nanoicl_to_nanoskl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoicl_to_nanoskl:
    if ele < lower_fence or ele > higher_fence :
    	nanoicl_to_nanoskl.remove(ele)
    	count = count + 1
    else :
    	new_df6.append(help_df6[count])
    	new_df5.append(help_df5[count])
    	#new_df6.append(new_df[6][count])
    	#new_df5.append(new_df[5][count])
    	count =count + 1
print("nanoicl_to_nanoskl")
print(len(nanoicl_to_nanoskl))
median_value = median(nanoicl_to_nanoskl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoicl_to_nanoskl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(nanoicl_to_nanoskl)))
print("geometrical mean ", sum(nanoicl_to_nanoskl)/len(nanoicl_to_nanoskl))
#cosine = np.dot(new_df6,new_df5)/(norm(new_df6)*norm(new_df5))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df6,new_df5)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df6,new_df5)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df6,new_df5))
mse = mean_squared_error(new_df6,new_df5)
print("mse:", mse)
sns.distplot(nanoicl_to_nanoskl, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with skylake values (nanobench)")
plt.savefig("nanoicl_to_nanoskl.png")
#plt.show()
#########################################nanoicl_to_nanoamd########################################################################################
nanoicl_to_nanoamd = []
help_df6 = []
help_df7 = []
new_df6 = []
new_df7 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[6][i] != 0 and new_df[7][i] != 0):
		help = float(new_df[6][i])/float(new_df[7][i])
		nanoicl_to_nanoamd.append(help)
		help_df6.append(new_df[6][i])
		help_df7.append(new_df[7][i])
		
q1, q3 = np.percentile(nanoicl_to_nanoamd, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoicl_to_nanoamd:
    if ele < lower_fence or ele > higher_fence :
    	nanoicl_to_nanoamd.remove(ele)
    	count = count + 1
    else :
    	new_df6.append(help_df6[count])
    	new_df7.append(help_df7[count])
    	#new_df6.append(new_df[6][count])
    	#new_df7.append(new_df[7][count])
    	count =count + 1
print("nanoicl_to_nanoamd")
print(len(nanoicl_to_nanoamd))
median_value = median(nanoicl_to_nanoamd)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoicl_to_nanoamd])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", sum(nanoicl_to_nanoamd)/len(nanoicl_to_nanoamd))
#print("geometrical mean ", str(statistics.geometric_mean(nanoicl_to_nanoamd)))
#cosine = np.dot(new_df6,new_df7)/(norm(new_df6)*norm(new_df7))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df6,new_df7)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df6,new_df7)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df6,new_df7))
mse = mean_squared_error(new_df6,new_df7)
print("mse:", mse)
sns.distplot(nanoicl_to_nanoamd, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake with AMD values (nanobench)")
plt.savefig("nanoicl_to_nanoamd.png")
#plt.show()
#########################################nanoamd_to_nanobdw########################################################################################
nanoamd_to_nanobdw = []
help_df7 = []
help_df4 = []
new_df7 = []
new_df4 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[7][i] != 0 and new_df[4][i] != 0):
		help = float(new_df[7][i])/float(new_df[4][i])
		nanoamd_to_nanobdw.append(help)
		help_df7.append(new_df[7][i])
		help_df4.append(new_df[4][i])
		
q1, q3 = np.percentile(nanoamd_to_nanobdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoamd_to_nanobdw:
    if ele < lower_fence or ele > higher_fence :
    	nanoamd_to_nanobdw.remove(ele)
    	count = count + 1
    else :
    	new_df7.append(help_df7[count])
    	new_df4.append(help_df4[count])
    	#new_df7.append(new_df[7][count])
    	#new_df4.append(new_df[4][count])
    	count =count + 1
print("nanoamd_to_nanobdw")
print(len(nanoamd_to_nanobdw))
median_value = median(nanoamd_to_nanobdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoamd_to_nanobdw])
print("median absolute",median_absolute_deviation)
#print("geometrical mean ", str(statistics.geometric_mean(nanoamd_to_nanobdw)))
print("geometrical mean ", sum(nanoamd_to_nanobdw)/len(nanoamd_to_nanobdw))
#cosine = np.dot(new_df7,new_df4)/(norm(new_df7)*norm(new_df4))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df7,new_df4)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df7,new_df4)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df7,new_df4))
mse = mean_squared_error(new_df7,new_df4)
print("mse:", mse)
sns.distplot(nanoamd_to_nanobdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare AMD with brodwell values (nanobench)")
plt.savefig("nanoamd_to_nanobdw.png")
#plt.show()
#########################################nanoamd_to_nanoskl########################################################################################
nanoamd_to_nanoskl = []
help_df7 = []
help_df5 = []
new_df7 = []
new_df5 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[7][i] != 0 and new_df[5][i] != 0):
		help = float(new_df[7][i])/float(new_df[5][i])
		nanoamd_to_nanoskl.append(help)
		help_df7.append(new_df[7][i])
		help_df5.append(new_df[5][i])
		
q1, q3 = np.percentile(nanoamd_to_nanoskl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoamd_to_nanoskl:
    if ele < lower_fence or ele > higher_fence :
    	nanoamd_to_nanoskl.remove(ele)
    	count = count + 1
    else :
    	new_df7.append(help_df7[count])
    	new_df5.append(help_df5[count])
    	#new_df7.append(new_df[7][count])
    	#new_df5.append(new_df[5][count])
    	count =count + 1
print("nanoamd_to_nanoskl")
print(len(nanoamd_to_nanoskl))
median_value = median(nanoamd_to_nanoskl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoamd_to_nanoskl])
print("median absolute",median_absolute_deviation)
#print("geometrical mean ", str(statistics.geometric_mean(nanoamd_to_nanoskl)))
print("geometrical mean ", sum(nanoamd_to_nanoskl)/len(nanoamd_to_nanoskl))
#cosine = np.dot(new_df7,new_df5)/(norm(new_df7)*norm(new_df5))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df7,new_df5)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df7,new_df5)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df7,new_df5))
mse = mean_squared_error(new_df7,new_df5)
print("mse:", mse)
sns.distplot(nanoamd_to_nanoskl, hist=True)
plt.xlim(0, 2)
plt.title("Compare AMD with skylake values (nanobench)")
plt.savefig("nanoamd_to_nanoskl.png")
#plt.show()
#########################################nanoamd_to_nanoicl########################################################################################
nanoamd_to_nanoicl = []
help_df7 = []
help_df6 = []
new_df7 = []
new_df6 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[7][i] != 0 and new_df[6][i] != 0):
		help = float(new_df[7][i])/float(new_df[6][i])
		nanoamd_to_nanoicl.append(help)
		help_df7.append(new_df[7][i])
		help_df6.append(new_df[6][i])
		
q1, q3 = np.percentile(nanoamd_to_nanoicl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in nanoamd_to_nanoicl:
    if ele < lower_fence or ele > higher_fence :
    	nanoamd_to_nanoicl.remove(ele)
    	count = count + 1
    else :
    	new_df7.append(help_df7[count])
    	new_df6.append(help_df6[count])
    	#new_df7.append(new_df[7][count])
    	#new_df6.append(new_df[6][count])
    	count =count + 1
print("nanoamd_to_nanoicl")
print(len(nanoamd_to_nanoicl))
median_value = median(nanoamd_to_nanoicl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in nanoamd_to_nanoicl])
print("median absolute",median_absolute_deviation)
#print("geometrical mean ", str(statistics.geometric_mean(nanoamd_to_nanoicl)))
print("geometrical mean ", sum(nanoamd_to_nanoicl)/len(nanoamd_to_nanoicl))
#cosine = np.dot(new_df7,new_df6)/(norm(new_df7)*norm(new_df6))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df7,new_df6)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df7,new_df6)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df7,new_df6))
mse = mean_squared_error(new_df7,new_df6)
print("mse:", mse)
sns.distplot(nanoamd_to_nanoicl, hist=True)
plt.xlim(0, 2)
plt.title("Compare AMD with icelake values (nanobench)")
plt.savefig("nanoamd_to_nanoicl.png")
#plt.show()
#########################################bdw_to_nanobdw########################################################################################
bdw_to_nanobdw = []
help_df0 = []
help_df4 = []
new_df0 = []
new_df4 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[0][i] != 0 and new_df[4][i] != 0):
		help = float(new_df[0][i])/float(new_df[4][i])
		bdw_to_nanobdw.append(help)
		help_df0.append(new_df[0][i])
		help_df4.append(new_df[4][i])
		
q1, q3 = np.percentile(bdw_to_nanobdw, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in bdw_to_nanobdw:
    if ele < lower_fence or ele > higher_fence :
    	bdw_to_nanobdw.remove(ele)
    	count = count + 1
    else :
    	new_df0.append(help_df0[count])
    	new_df4.append(help_df4[count])
    	#new_df0.append(new_df[0][count])
    	#new_df4.append(new_df[4][count])
    	count =count + 1
print("bdw_to_nanobdw")
print(len(bdw_to_nanobdw))
median_value = median(bdw_to_nanobdw)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in bdw_to_nanobdw])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(bdw_to_nanobdw)))
#cosine = np.dot(new_df0,new_df4)/(norm(new_df0)*norm(new_df4))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df0,new_df4)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df0,new_df4)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df0,new_df4))
mse = mean_squared_error(new_df0,new_df4)
print("mse:", mse)
sns.distplot(bdw_to_nanobdw, hist=True)
plt.xlim(0, 2)
plt.title("Compare brodwell values with timming-harness and nanobench")
plt.savefig("bdw_to_nanobdw.png")
#plt.show()
#########################################skl_to_nanoskl########################################################################################
skl_to_nanoskl = []
help_df2 = []
help_df5 = []
new_df2 = []
new_df5 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[2][i] != 0 and new_df[5][i] != 0):
		help = float(new_df[2][i])/float(new_df[5][i])
		skl_to_nanoskl.append(help)
		help_df2.append(new_df[2][i])
		help_df5.append(new_df[5][i])
		
q1, q3 = np.percentile(skl_to_nanoskl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in skl_to_nanoskl:
    if ele < lower_fence or ele > higher_fence :
    	skl_to_nanoskl.remove(ele)
    	count = count + 1
    else :
    	new_df2.append(help_df2[count])
    	new_df5.append(help_df5[count])
    	#new_df2.append(new_df[2][count])
    	#new_df5.append(new_df[5][count])
    	count =count + 1
print("skl_to_nanoskl")
print(len(skl_to_nanoskl))
median_value = median(skl_to_nanoskl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in skl_to_nanoskl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(skl_to_nanoskl)))
#cosine = np.dot(new_df2,new_df5)/(norm(new_df2)*norm(new_df5))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df2,new_df5)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df2,new_df5)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df2,new_df5))
mse = mean_squared_error(new_df2,new_df5)
print("mse:", mse)
sns.distplot(skl_to_nanoskl, hist=True)
plt.xlim(0, 2)
plt.title("Compare skylake values with timming-harness and nanobench")
plt.savefig("skl_to_nanoskl.png")
#plt.show()
#########################################icl_to_nanoicl########################################################################################
icl_to_nanoicl = []
help_df3 = []
help_df6 = []
new_df3 = []
new_df6 = []
q1 = 0
q3 = 0
count = 0
for i in range (209496):
	if (new_df[3][i] != 0 and new_df[6][i] != 0):
		help = float(new_df[3][i])/float(new_df[6][i])
		icl_to_nanoicl.append(help)
		help_df3.append(new_df[3][i])
		help_df6.append(new_df[6][i])
		
q1, q3 = np.percentile(icl_to_nanoicl, [25,75])
iqr = q3 - q1
lower_fence = q1 - (1.5*iqr)
higher_fence = q3 + (1.5*iqr)
for ele in icl_to_nanoicl:
    if ele < lower_fence or ele > higher_fence :
    	icl_to_nanoicl.remove(ele)
    	count = count + 1
    else :
    	new_df3.append(help_df3[count])
    	new_df6.append(help_df6[count])
    	#new_df3.append(new_df[3][count])
    	#new_df6.append(new_df[6][count])
    	count =count + 1
print("icl_to_nanoicl")
print(len(icl_to_nanoicl))
median_value = median(icl_to_nanoicl)
print("median ",median_value)
median_absolute_deviation = median([abs(number-median_value) for number in icl_to_nanoicl])
print("median absolute",median_absolute_deviation)
print("geometrical mean ", str(statistics.geometric_mean(icl_to_nanoicl)))
#cosine = np.dot(new_df3,new_df6)/(norm(new_df3)*norm(new_df6))
#print("cosine:",cosine)
#eu = euclidean_distance(new_df3,new_df6)
#print("euclidean distance:",eu)
tau, p_value = stats.kendalltau(new_df3,new_df6)
print("tau ", tau)
print("p_value ", p_value)
print("r2_score ",r2_score(new_df3,new_df6))
mse = mean_squared_error(new_df3,new_df6)
print("mse:", mse)
sns.distplot(icl_to_nanoicl, hist=True)
plt.xlim(0, 2)
plt.title("Compare icelake values with timming-harness and nanobench")
plt.savefig("icl_to_nanoicl.png")
#plt.show()
###########################################HEATMAPS#############################################################################################
'''
l2 = new_df[[0, 1]]
corr = l2.corr()
x_axis_labels = ["BDW", "HSW"] 
y_axis_labels = ["BDW", "HSW"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with haswell values")
plt.savefig("heatmap_bdw_to_hsw.png")
plt.show()

l2 = new_df[[0, 2]]
corr = l2.corr()
x_axis_labels = ["BDW", "SKL"] 
y_axis_labels = ["BDW", "SKL"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with skylake values")
plt.savefig("heatmap_bdw_to_skl.png")
plt.show()

l2 = new_df[[0, 3]]
corr = l2.corr()
x_axis_labels = ["BDW", "ICL"] 
y_axis_labels = ["BDW", "ICL"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with icelake values")
plt.savefig("heatmap_bdw_to_icl.png")
plt.show()

l2 = new_df[[1, 2]]
corr = l2.corr()
x_axis_labels = ["HSW", "SKL"] 
y_axis_labels = ["HSW", "SKL"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing haswell with skylake values")
plt.savefig("heatmap_hsw_to_skl.png")
plt.show()

l2 = new_df[[1, 3]]
corr = l2.corr()
x_axis_labels = ["HSW", "ICL"] 
y_axis_labels = ["HSW", "ICL"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing haswell with icelake values")
plt.savefig("heatmap_hsw_to_icl.png")
plt.show()

l2 = new_df[[2, 3]]
corr = l2.corr()
x_axis_labels = ["SKL", "ICL"] 
y_axis_labels = ["SKL", "ICL"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing skylake with icelake values")
plt.savefig("heatmap_skl_to_icl.png")
plt.show()

l2 = new_df[[4, 5]]
corr = l2.corr()
x_axis_labels = ["BDW(Nano)", "SKL(Nano)"] 
y_axis_labels = ["BDW(Nano)", "SKL(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with skylake values (nanobench)")
plt.savefig("heatmap_nanobdw_to_nanoskl.png")
plt.show()

l2 = new_df[[4, 6]]
corr = l2.corr()
x_axis_labels = ["BDW(Nano)", "ICL(Nano)"] 
y_axis_labels = ["BDW(Nano)", "ICL(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with icelake values (nanobench)")
plt.savefig("heatmap_nanobdw_to_nanoicl.png")
plt.show()

l2 = new_df[[4, 7]]
corr = l2.corr()
x_axis_labels = ["BDW(Nano)", "AMD(Nano)"] 
y_axis_labels = ["BDW(Nano)", "AMD(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with AMD values (nanobench)")
plt.savefig("heatmap_nanobdw_to_nanoamd.png")
plt.show()

l2 = new_df[[5, 6]]
corr = l2.corr()
x_axis_labels = ["SKL(Nano)", "ICL(Nano)"] 
y_axis_labels = ["SKL(Nano)", "ICL(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing skylake with icelake values (nanobench)")
plt.savefig("heatmap_nanoskl_to_nanoicl.png")
plt.show()

l2 = new_df[[5, 7]]
corr = l2.corr()
x_axis_labels = ["SKL(Nano)", "AMD(Nano)"] 
y_axis_labels = ["SKL(Nano)", "AMD(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing skylake with AMD values (nanobench)")
plt.savefig("heatmap_nanoskl_to_nanoamd.png")
plt.show()

l2 = new_df[[6, 7]]
corr = l2.corr()
x_axis_labels = ["ICL(Nano)", "AMD(Nano)"] 
y_axis_labels = ["ICL(Nano)", "AMD(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing icelake with AMD values (nanobench)")
plt.savefig("heatmap_nanoicl_to_nanoamd.png")
plt.show()

l2 = new_df[[0, 4]]
corr = l2.corr()
x_axis_labels = ["BDW", "BDW(Nano)"] 
y_axis_labels = ["BDW", "BDW(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing brodwell with timming-harness and nanobench")
plt.savefig("heatmap_bdw_to_nanobdw.png")
plt.show()

l2 = new_df[[2, 5]]
corr = l2.corr()
x_axis_labels = ["SKL", "SKL(Nano)"] 
y_axis_labels = ["SKL", "SKL(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing skylake with timming-harness and nanobench")
plt.savefig("heatmap_skl_to_nanoskl.png")
plt.show()

l2 = new_df[[3, 6]]
corr = l2.corr()
x_axis_labels = ["ICL", "ICL(Nano)"] 
y_axis_labels = ["ICL", "ICL(Nano)"] 
sns.heatmap(corr, cmap='RdBu', vmin=-1, vmax=1, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.title("Heatmap comparing icelake with timming-harness and nanobench")
plt.savefig("heatmap_icl_to_nanoicl.png")
plt.show()
################################################################################################################################
'''
