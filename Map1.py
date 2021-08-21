import sys

# input comes from STDIN (standard input)

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split("\t")
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # this code was modified by Rupesh
        #now this code has to be verified by binesh jose
        print (word)
