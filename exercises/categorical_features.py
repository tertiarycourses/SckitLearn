import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]

print(vec.fit_transform(measurements).toarray())

print(vec.get_feature_names())