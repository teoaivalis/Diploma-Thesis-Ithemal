from collections import defaultdict
import csv
import numpy as np
contents1 = []
contents2 = []
contents3 = []
contents4 = []
contents5 = []
contents6 = []
contents7 = []
contents8 = []
contents9 = []
contents10 = []
contents11 = []
contents12 = []
contents13 = []
contents14 = []	
contents15 = []
contents16 = []

with open('/home/delluser/Documents/ithemal/Ithemal/throughput_bdw.csv','r') as bdw:
	csvreader = csv.reader(bdw)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents1.append(row[0])
			count = count + 1
		else:
			contents2.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_hsw.csv','r') as hsw:
	csvreader = csv.reader(hsw)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents3.append(row[0])
			count = count + 1
		else:
			contents4.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_skl.csv','r') as skl:
	csvreader = csv.reader(skl)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents5.append(row[0])
			count = count + 1
		else:
			contents6.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/throughput_icl.csv','r') as icl:
	csvreader = csv.reader(icl)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents7.append(row[0])
			count = count + 1
		else:
			contents8.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_bdw.csv','r') as nano_bdw:
	csvreader = csv.reader(nano_bdw)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents9.append(row[0])
			count = count + 1
		else:
			contents10.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_skl.csv','r') as nano_skl:
	csvreader = csv.reader(nano_skl)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents11.append(row[0])
			count = count + 1
		else:
			contents12.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_icl.csv','r') as nano_icl:
	csvreader = csv.reader(nano_icl)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents13.append(row[0])
			count = count + 1
		else:
			contents14.append(row[0])
			count = count + 1
with open('/home/delluser/Documents/ithemal/Ithemal/nanothroughput_amd.csv','r') as nano_amd:
	csvreader = csv.reader(nano_amd)
	count = 0
	for row in csvreader:
		if (count % 2 == 0):
			contents15.append(row[0])
			count = count + 1
		else:
			contents16.append(row[0])
			count = count + 1
l1 = []
l3 = []
l1 = (list(set(contents1).difference(contents3)))
l3 = (list(set(contents3).difference(contents1)))
for i in l1:
        contents3.append(i)
for i in l3:
        contents1.append(i)
for i in range(len(l3)):
        contents2.append("0")
for i in range(len(l1)):
        contents4.append("0")


l1 = []
l5 = []
l1 = (list(set(contents1).difference(contents5)))
l5 = (list(set(contents5).difference(contents1)))

for i in l1:
        contents5.append(i)
for i in l5:
        contents1.append(i)
for i in range(len(l5)):
        contents2.append("0")
for i in range(len(l1)):
        contents6.append("0")


l1 = []
l7 = []
l1 = (list(set(contents1).difference(contents7)))
l7 = (list(set(contents7).difference(contents1)))

for i in l1:
        contents7.append(i)
for i in l7:
        contents1.append(i)
for i in range(len(l7)):
        contents2.append("0")
for i in range(len(l1)):
        contents8.append("0")

l1 = []
l9 = []
l1 = (list(set(contents1).difference(contents9)))
l9 = (list(set(contents9).difference(contents1)))

for i in l1:
        contents9.append(i)
for i in l9:
        contents1.append(i)
for i in range(len(l9)):
        contents2.append("0")
for i in range(len(l1)):
        contents10.append("0")

l1 = []
l11 = []
l1 = (list(set(contents1).difference(contents11)))
l11 = (list(set(contents11).difference(contents1)))

for i in l1:
        contents11.append(i)
for i in l11:
        contents1.append(i)
for i in range(len(l11)):
        contents2.append("0")
for i in range(len(l1)):
        contents12.append("0")

l1 = []
l13 = []
l1 = (list(set(contents1).difference(contents13)))
l11 = (list(set(contents13).difference(contents1)))

for i in l1:
        contents13.append(i)
for i in l13:
        contents1.append(i)
for i in range(len(l13)):
        contents2.append("0")
for i in range(len(l1)):
        contents14.append("0")

l1 = []
l15 = []
l1 = (list(set(contents1).difference(contents15)))
l11 = (list(set(contents15).difference(contents1)))

for i in l1:
        contents15.append(i)
for i in l15:
        contents1.append(i)
for i in range(len(l15)):
        contents2.append("0")
for i in range(len(l1)):
        contents16.append("0")

l3 = []
l5 = []
l3 = (list(set(contents3).difference(contents5)))
l5 = (list(set(contents5).difference(contents3)))

for i in l3:
        contents5.append(i)
for i in l5:
        contents3.append(i)
for i in range(len(l5)):
        contents4.append("0")
for i in range(len(l3)):
        contents6.append("0")

l3 = []
l7 = []
l3 = (list(set(contents3).difference(contents7)))
l7 = (list(set(contents7).difference(contents3)))

for i in l3:
        contents7.append(i)
for i in l7:
        contents3.append(i)
for i in range(len(l7)):
        contents4.append("0")
for i in range(len(l3)):
        contents8.append("0")

l3 = []
l9 = []
l3 = (list(set(contents3).difference(contents9)))
l9 = (list(set(contents9).difference(contents3)))

for i in l3:
        contents9.append(i)
for i in l9:
        contents3.append(i)
for i in range(len(l9)):
        contents4.append("0")
for i in range(len(l3)):
        contents10.append("0")

l3 = []
l11 = []
l3 = (list(set(contents3).difference(contents11)))
l11 = (list(set(contents11).difference(contents3)))

for i in l3:
        contents11.append(i)
for i in l11:
        contents3.append(i)
for i in range(len(l11)):
        contents4.append("0")
for i in range(len(l3)):
        contents12.append("0")

l3 = []
l13 = []
l3 = (list(set(contents3).difference(contents13)))
l13 = (list(set(contents13).difference(contents3)))

for i in l3:
        contents13.append(i)
for i in l13:
        contents3.append(i)
for i in range(len(l13)):
        contents4.append("0")
for i in range(len(l3)):
        contents14.append("0")

l3 = []
l15 = []
l3 = (list(set(contents3).difference(contents15)))
l15 = (list(set(contents15).difference(contents3)))

for i in l3:
        contents15.append(i)
for i in l15:
        contents3.append(i)
for i in range(len(l15)):
        contents4.append("0")
for i in range(len(l3)):
        contents16.append("0")

l5 = []
l7 = []
l5 = (list(set(contents5).difference(contents7)))
l7 = (list(set(contents7).difference(contents5)))

for i in l5:
        contents7.append(i)
for i in l7:
        contents5.append(i)
for i in range(len(l7)):
        contents6.append("0")
for i in range(len(l5)):
        contents8.append("0")

l5 = []
l9 = []
l5 = (list(set(contents5).difference(contents9)))
l9 = (list(set(contents9).difference(contents5)))

for i in l5:
        contents9.append(i)
for i in l9:
        contents5.append(i)
for i in range(len(l9)):
        contents6.append("0")
for i in range(len(l5)):
        contents10.append("0")

l5 = []
l11 = []
l5 = (list(set(contents5).difference(contents11)))
l11 = (list(set(contents11).difference(contents5)))

for i in l5:
        contents11.append(i)
for i in l11:
        contents5.append(i)
for i in range(len(l11)):
        contents6.append("0")
for i in range(len(l5)):
        contents12.append("0")

l5 = []
l13 = []
l5 = (list(set(contents5).difference(contents13)))
l13 = (list(set(contents13).difference(contents5)))

for i in l5:
        contents13.append(i)
for i in l13:
        contents5.append(i)
for i in range(len(l13)):
        contents6.append("0")
for i in range(len(l5)):
        contents14.append("0")

l5 = []
l15 = []
l5 = (list(set(contents5).difference(contents15)))
l15 = (list(set(contents15).difference(contents5)))

for i in l5:
        contents15.append(i)
for i in l15:
        contents5.append(i)
for i in range(len(l15)):
        contents6.append("0")
for i in range(len(l5)):
        contents16.append("0")

l7 = []
l9 = []
l7 = (list(set(contents5).difference(contents7)))
l9 = (list(set(contents7).difference(contents5)))

for i in l7:
        contents9.append(i)
for i in l9:
        contents7.append(i)
for i in range(len(l9)):
        contents8.append("0")
for i in range(len(l7)):
        contents10.append("0")

l7 = []
l11 = []
l7 = (list(set(contents7).difference(contents11)))
l11 = (list(set(contents11).difference(contents7)))

for i in l7:
        contents11.append(i)
for i in l11:
        contents7.append(i)
for i in range(len(l11)):
        contents8.append("0")
for i in range(len(l7)):
        contents12.append("0")

l7 = []
l13 = []
l7 = (list(set(contents7).difference(contents13)))
l13 = (list(set(contents13).difference(contents7)))

for i in l7:
        contents13.append(i)
for i in l13:
        contents7.append(i)
for i in range(len(l13)):
        contents8.append("0")
for i in range(len(l7)):
        contents14.append("0")

l7 = []
l15 = []
l7 = (list(set(contents7).difference(contents15)))
l15 = (list(set(contents15).difference(contents7)))

for i in l7:
        contents15.append(i)
for i in l15:
        contents7.append(i)
for i in range(len(l15)):
        contents8.append("0")
for i in range(len(l7)):
        contents16.append("0")

l9 = []
l11 = []
l9 = (list(set(contents9).difference(contents11)))
l11 = (list(set(contents11).difference(contents9)))

for i in l9:
        contents11.append(i)
for i in l11:
        contents9.append(i)
for i in range(len(l11)):
        contents10.append("0")
for i in range(len(l9)):
        contents12.append("0")

l9 = []
l13 = []
l9 = (list(set(contents9).difference(contents13)))
l13 = (list(set(contents13).difference(contents9)))

for i in l9:
        contents13.append(i)
for i in l13:
        contents9.append(i)
for i in range(len(l13)):
        contents10.append("0")
for i in range(len(l9)):
        contents14.append("0")

l9 = []
l15 = []
l9 = (list(set(contents9).difference(contents15)))
l15 = (list(set(contents15).difference(contents9)))

for i in l9:
        contents15.append(i)
for i in l15:
        contents9.append(i)
for i in range(len(l15)):
        contents10.append("0")
for i in range(len(l9)):
        contents16.append("0")

l11 = []
l13 = []
l11 = (list(set(contents11).difference(contents13)))
l13 = (list(set(contents13).difference(contents11)))

for i in l11:
        contents13.append(i)
for i in l13:
        contents11.append(i)
for i in range(len(l13)):
        contents12.append("0")
for i in range(len(l11)):
        contents14.append("0")

l11 = []
l15 = []
l11 = (list(set(contents11).difference(contents15)))
l15 = (list(set(contents15).difference(contents11)))

for i in l11:
        contents15.append(i)
for i in l15:
        contents11.append(i)
for i in range(len(l15)):
        contents12.append("0")
for i in range(len(l11)):
        contents16.append("0")

l13 = []
l15 = []
l13 = (list(set(contents13).difference(contents15)))
l15 = (list(set(contents15).difference(contents13)))

for i in l13:
        contents15.append(i)
for i in l15:
        contents13.append(i)
for i in range(len(l15)):
        contents14.append("0")
for i in range(len(l13)):
        contents16.append("0")


bdw_dict = {contents1[i]: contents2[i] for i in range(len(contents1))}
hsw_dict = {contents3[i]: contents4[i] for i in range(len(contents3))}
skl_dict = {contents5[i]: contents6[i] for i in range(len(contents5))}
icl_dict = {contents7[i]: contents8[i] for i in range(len(contents7))}
nano_bdw_dict = {contents9[i]: contents10[i] for i in range(len(contents9))}
nano_skl_dict = {contents11[i]: contents12[i] for i in range(len(contents11))}
nano_icl_dict = {contents13[i]: contents14[i] for i in range(len(contents13))}
nano_amd_dict = {contents15[i]: contents16[i] for i in range(len(contents15))}


dd = defaultdict(list)
for d in (bdw_dict, hsw_dict, skl_dict, icl_dict, nano_bdw_dict, nano_skl_dict, nano_icl_dict, nano_amd_dict): 
    for key, value in d.items():
        dd[key].append(float((value)))
        

keysList = list(dd.keys())
with open("merge_keys.csv", 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(keysList)

#with open("merge_values.csv", "w") as outfile:
#   writer = csv.writer(outfile)
#   writer.writerow(dd.keys())
#   writer.writerows(zip(*dd.values()))

