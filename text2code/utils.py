from typing import List, Dict, Any
from torch.utils.data import Dataset
import json

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(self.data[idx])

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the jsonl file.

    Returns:
        List[Dict[str, Any]]: A list where each element is a dictionary
                              representing a JSON object from the file.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data