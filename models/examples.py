import json 

with open('attack_dict.json', 'r') as f:
    attack_dict = json.load(f)    # returns dict
    # print(type(attack_dict))

for item in attack_dict.items():
    # item[0] returns the object type, item[1] returns info about it
    examples = item[1]['example_uses']
    if examples:
        print("Object type:", item[0])
        print("Technique name:", item[1]['name'])
        print("Example_uses:", examples, '\n')
# for k in attack_dict.items():
#     examples = k[1]['name']
#     print(examples)
