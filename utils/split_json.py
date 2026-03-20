import sys
import json

with open('log_algo_multi.txt', "r") as f:
    json_in = json.loads(f.readline())

json_first = {}
json_second = {}
for k in json_in:
    json_first[k] = [i for i in json_in[k][:200]]
    json_second[k] = [i for i in json_in[k][200:400]]

json_first = json.dumps(json_first)
json_second = json.dumps(json_second)

with open(sys.argv[1], "w") as f:
    f.write(json_first)

with open(sys.argv[2], "w") as f:
    f.write(json_second)
