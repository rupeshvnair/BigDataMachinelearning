import sys

newval = []

for line in sys.stdin:
    value = line.strip()
    value1 = value.split()
    newval = newval + value1

newval1 = list(set(newval))

for word in newval1:
    count = 0
    for values in newval:
        if word == values:
            count = count +1
    print ('%s\t%s' % (word, count))







