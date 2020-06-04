import json
import sys

res = json.load(open(sys.argv[1], "r"))

print(len(res['annotations']))
