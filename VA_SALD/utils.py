import inspect
import torch
from typing import List, Union
import contextlib

@contextlib.contextmanager
def on_device(model, device):
    original_device = model.device
    model.to(device)
    try:
        yield
    finally:
        model.to(original_device)

def offload_model(model):
    original_device = model.device
    model.to("cpu")
    try:
        yield
    finally:
        model.to(original_device)
class collect_calls:
    def __init__(self, target_method, arg_names=None):
        self.obj, self.func_name = target_method.__self__, target_method.__name__
        self.original_func = getattr(self.obj, self.func_name)
        self.param_names = list(inspect.signature(self.original_func).parameters)
        self.names_to_collect = set(arg_names or [])
        self.data = []

    def __enter__(self):
        def wrapper(*args, **kwargs):
            res = self.original_func(*args, **kwargs)
            all_args = {**dict(zip(self.param_names, args)), **kwargs}
            self.data.append({'args': {k: v for k, v in all_args.items() if k in self.names_to_collect}, 'return': res})
            return res
        setattr(self.obj, self.func_name, wrapper)
        return self.data

    def __exit__(self, *args):
        setattr(self.obj, self.func_name, self.original_func)


def batches_dict(data, batch_size):
    n = len(next(iter(data.values())))
    for i in range(0, n, batch_size):
        yield {k: v[i:i+batch_size] for k, v in data.items()}

def concat(data: Union[List[torch.Tensor], List[List]]):
    if isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    elif isinstance(data[0], list):
        return sum(data, [])
    else:
        raise ValueError(f"Unsupported data type: {type(data[0])}")

def batch_generator(args_dict, batch_size, total_len, keys_to_slice):
    """Yields sliced dictionaries for each batch."""
    for i in range(0, total_len, batch_size):
        chunk = {}
        for k, v in args_dict.items():
            if k in keys_to_slice:
                chunk[k] = v[i : i + batch_size]
            else:
                chunk[k] = v # Keep static arguments as is
        yield chunk

"""
Context manager that intercepts method calls to automatically chunk large inputs 
(lists/tensors) into mini-batches to prevent OOM, executes them sequentially, 
and concatenates the results back into a single output.
"""
class auto_batching:
    def __init__(self, target_method, batch_size):
        self.obj = target_method.__self__
        self.func_name = target_method.__name__
        self.original_func = target_method
        self.batch_size = batch_size
        self.signature = inspect.signature(target_method)

    def __enter__(self):
        def wrapper(*args, **kwargs):
            # 1. Bind all args to names to handle *args and **kwargs uniformly
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_args = bound_args.arguments

            # 2. Detect Batch Size (N) and Identify Sliced Keys
            N = None
            keys_to_slice = set()

            for name, val in all_args.items():
                # We interpret Lists and Tensors as batchable data
                if isinstance(val, (list, torch.Tensor)):
                    current_len = len(val)
                    if N is None:
                        N = current_len
                    elif current_len != N:
                        # Enforce strict length consistency as requested
                        raise ValueError(f"Batch dimension mismatch! Arg '{name}' has len {current_len}, expected {N}.")
                    
                    keys_to_slice.add(name)
            
            # If no iterable args found, just run once
            if N is None:
                return self.original_func(*args, **kwargs)

            # 3. Chunk, Execute, Collect
            outputs = []
            # Reuse logic similar to batches_dict via helper
            for chunk_kwargs in batch_generator(all_args, self.batch_size, N, keys_to_slice):
                res = self.original_func(**chunk_kwargs)
                outputs.append(res)

            # 4. Concatenate and Return
            return concat(outputs)

        # Monkey patch
        setattr(self.obj, self.func_name, wrapper)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original
        setattr(self.obj, self.func_name, self.original_func)

class TensorIndexableList(list):
    def __getitem__(self, key):
        if isinstance(key, torch.Tensor) and key.dim() == 1:
            return [super(TensorIndexableList, self).__getitem__(i) for i in key.tolist()]
        elif isinstance(key, list):
            return [super(TensorIndexableList, self).__getitem__(i) for i in key]
        else:
            return super().__getitem__(key)
