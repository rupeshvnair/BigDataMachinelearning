import sys

newval = []

for line in sys.stdin:
    value = line.split(",")
    newval = newval + value

newval1 = list(set(newval))

for word in newval1:
    count = 0
    for values in newval:
        if word == values:
            count = count +1
    print(word,count)







