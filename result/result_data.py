list1 = []
list2 = []
path = 'result.txt'
out_path = 'result1.txt'
in_file = open(path)
out_file = open(out_path, 'w')
for line in in_file.readlines():
	res = line.split(',')
	res2 = int(res[1])
	res2 += 1
	fix_res = str(res2) + '\r'
	out_file.write(fix_res)




