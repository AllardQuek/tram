import pickle

with open('model_dict.p', 'rb') as handle:
    result = pickle.load(handle)
print(type(result))
