import json
import sys
import glob

total = 0
for fp in glob.glob(f'{sys.argv[1]}/*.json'):
    total += len(json.load(open(fp, 'r')))

print(total)