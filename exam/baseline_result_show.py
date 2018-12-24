from prettytable import PrettyTable
import glob
import re
import os
from collections import defaultdict

result = defaultdict(lambda : defaultdict(float))

pattern = re.compile("new users: RMSE ([^ ]*)")
for fn in glob.glob("log/baseline/*"):
    with open(fn) as f:
        try:
            rmse = pattern.findall(f.read())[0]
        except IndexError:
            continue
        f = os.path.basename(fn)
        domain, method, _ = f.split(".")
        result[domain][method] = rmse

result = {k: dict(v) for k, v in result.items()}

columns = result.values()[0].keys()
table = ["domain"] + columns
table = PrettyTable(table)
for k, v in result.items():
    try:
        table.add_row([k] + [v[m] for m in columns])
    except KeyError:
        continue

print table
