import json

with open('example.json', 'r') as f:
    json_string = f.read()
    print(len(json_string))
    json_object = json.loads(json_string)

json_object["p1"]['action'] = 'Pootis'

with open('example.json', 'w') as f:
    f.write(json.dumps(json_object))

