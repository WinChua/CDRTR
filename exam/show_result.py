import pickle
import math
import sys
from prettytable import PrettyTable
result = PrettyTable(["domain", "rmse"])


def get_rmse(result_file):
    with open(result_file) as f:
        data = pickle.load(f)
    domain = result_file.split("/")[-2].split("_")[0]
    return domain, math.sqrt(min(data))


Usage = '''
<Usage %s test_mse.pk_path ...>
e.g:
    python show_result.py `find . -name "test*.pk"`'''

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print Usage % sys.argv[0]
        sys.exit(1)

    for res in sys.argv[1:]:
        result.add_row(get_rmse(res))

    print result
