import sys
import json

with open('log_algo.txt', "r") as f:
    json_in = json.loads(f.readline())

json_empty = {}
for k in json_in:
    json_empty[k] = []
json_empty = json.dumps(json_empty)

with open(sys.argv[1], "w") as f:
    f.write(json_empty)
    
