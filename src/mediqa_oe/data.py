import os
import json
from typing import Optional, Union

import pandas as pd
from datasets import load_dataset, Dataset


class MedicalOrderDataLoader:
    def __init__(self, trs_json_path: str):
        if not os.path.exists(trs_json_path):
            raise FileNotFoundError(f"Dataset file not found: {trs_json_path}")
        
        self._validate_json_structure(trs_json_path)

        self.ds = load_dataset("json", data_files=trs_json_path, field="train", split="train")
        self.ds_val = load_dataset("json", data_files=trs_json_path, field="dev", split="train")


    def _validate_json_structure(self, trs_json_path: str) -> None:
        with open(trs_json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON root should be a dictionary")
        
        required_fields = ['train', 'dev']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        for split_name, split_data in data.items():
            if not isinstance(split_data, list):
                raise ValueError(f"Split '{split_name}' should be a list, got {type(split_data)}")

    def get_pandas(self) -> Union[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if isinstance(self.ds, Dataset) and isinstance(self.ds_val, Dataset):
            return self.ds.to_pandas(), self.ds_val.to_pandas() # type: ignore
        