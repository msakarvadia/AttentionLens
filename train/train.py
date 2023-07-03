import sys
sys.path.append('../')
from data.get_data import get_data

dataloader = get_data(batch_size=16)

for data in dataloader:
    print(data['text'])
    break
