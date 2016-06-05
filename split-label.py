label_file = open('label_file.txt', 'r').readline().split(',')
output = open('splitted_labels', 'w')
for l in label_file:
	output.write(l)
	output.write('\n')
	