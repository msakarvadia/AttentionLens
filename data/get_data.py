import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_data(device, streaming=True, pin_memory=True, batch_size=32, dataset_name="c4", num_workers=16):
    # will start loading the data when iterated over
    #data  = load_dataset(dataset_name, split="train", streaming=streaming)  

    #TODO (MS): discuss with team about whether or not to preprocess batches to ensure all items have same length
    data  = load_dataset(dataset_name, 'en', split="train", streaming=streaming)  
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                             pin_memory=pin_memory, pin_memory_device=device, num_workers=num_workers)

    return dataloader
