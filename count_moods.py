from collections import Counter
import json

data = json.load(open('new_training_data.json'))

counts = Counter(data.values())
print(counts)