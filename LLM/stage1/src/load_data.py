import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt) #tokenizes entire text, and converts them into token IDs as a single step.
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.max_length = max_length
        self.stride = stride
        self.n_windows = (len(token_ids) - self.max_length) // self.stride

    def __len__(self):
        return self.n_windows # returns the total number of rows in the dataset

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.token_ids[start : start + self.max_length + 1]
        return chunk[:-1].clone(), chunk[1:].clone()

def create_dataloader(txt,
                      tokenizer=None,
                      enc_name = 'gpt2',
                      batch_size=4,
                      max_length=256,
                      stride=128,
                      shuffle=True,
                      drop_last=True,
                      num_workers=0):
    tokenizer = tokenizer or tiktoken.get_encoding(enc_name)
    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last, #drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
                      num_workers=num_workers) # num of CPU processes to use for preprocessing.

class StreamingGPTDataset(IterableDataset):
    """
    if we ever iterate once per epoch and do not need random indexing, we can yield windows on the fly.
    this would avoid storing the full token list in a single tensor.
    we could still wrap it in a Dataloder, removing shuffle and relying on the object's own batching.
    """
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.tokens = tokenizer.encode(txt)
        self.max_length = max_length
        self.stride = stride

    def __iter__(self):
        # simple generator over sliding windows
        for i in range(0, len(self.tokens) - self.max_length, self.stride):
            chunk = self.tokens[i : i + self.max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],  dtype=torch.long)
            yield x, y

