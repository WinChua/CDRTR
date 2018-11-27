import pickle
import math
import sys
import os
with open(sys.argv[1]) as f:
    data = pickle.load(f)
domain = sys.argv[1].split("/")[-2].split("_")[0]
print(domain, "rmse is", math.sqrt(min(data)))
