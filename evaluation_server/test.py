import json

with open('example.json', 'r') as f:
    json_string = f.read()
    print(len(json_string))
    json_object = json.loads(json_string)

x = json_object["P1"]['action']
encoded_x = x.encode()
print(encoded_x)
decoded_x = encoded_x.decode('utf-8')
print(decoded_x)
print(len(encoded_x))





