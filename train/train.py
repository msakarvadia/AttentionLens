import sys
sys.path.append('../')
from data.get_data import get_data
from model.get_model import get_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

#TODO fix device for dataloader
dataloader = get_data(batch_size=16)

model = get_model(device=device)

for data in dataloader:
    print(data['text'])
    break
