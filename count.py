
f = open("processed_feature.txt", "r")

for line in f:
	string = line.split("^")
	print (string)
	print ("length is", len(string))