import sys
import numpy as np


def inspect():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    infile = open(input_path, "r", encoding="UTF-8")
    outfile = open(output_path, "w", encoding="UTF-8")
    data = infile.readlines()
    infile.close()
    tot = len(data) - 1
    count = dict()
    for s in data[1:]:
        typ = s.split('\t')[-1]
        if typ not in count:
            count[typ] = 1
        else:
            count[typ] += 1
    entropy = 0
    error = 1
    majorCnt = 0
    for ele in count:
        entropy -= count[ele] / tot * np.log2(count[ele] / tot)
        majorCnt = max(majorCnt, count[ele])
    error -= majorCnt / tot
    outfile.write("entropy: %.12f\n" % entropy)
    outfile.write("error: %.12f\n" % error)
    outfile.close()
    return entropy, error


if __name__ == "__main__":
    inspect()
    # print(inspect())
