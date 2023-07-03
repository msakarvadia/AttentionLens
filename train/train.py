import sys
sys.path.append('../')
from data.get_data import get_data
from model.get_model import get_model
from lense.get_lense import get_lense
import torch


device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory=False
if device!="cpu":
    pin_memory=True

dataloader = get_data(batch_size=2, pin_memory=pin_memory, device=device)

model = get_model(device=device)

lense = get_lense()

for data in dataloader:
    print(model(data['text']))
    print(data['text'])
    break
