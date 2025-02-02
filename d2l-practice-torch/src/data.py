from d2l import torch as d2l
import torch
from dataclasses import dataclass
import random
import sys
print(sys.executable)

class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

@dataclass
class SyntheticRegressionData(DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

    
    def get_dataloader(self, train):
        """
        Generate a data loader for training or validation.

        Args:
            train (bool): Whether to generate a data loader for training. If False, generate a data loader for validation.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        # Determine the slice indices based on whether it's for training or validation
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        # Call the get_tensorloader method to create and return a DataLoader
        return self.get_tensorloader((self.X, self.y), train, i)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """
        Create a data loader from a tuple of tensors.

        Args:
            tensors (tuple): A tuple of tensors.
            train (bool): Whether the data loader is for training.
            indices (slice): A slice object indicating which indices to use.

        Returns:
            DataLoader: A PyTorch DataLoader object.
        """
        # Slice the tensors according to the provided indices
         # 设置断点
        tensors = tuple(a[indices] for a in tensors)

        # Create a TensorDataset from the sliced tensors (* for unpacking)
        # TensorDataset combines tensors into a single dataset for efficient access
        dataset = torch.utils.data.TensorDataset(*tensors)

        # Create and return a DataLoader from the dataset
        # DataLoader provides several key benefits:
        # 1. Automatic batching of data
        # 2. Shuffling of data when training (if shuffle=True)
        # 3. Parallel data loading with multiple workers
        # 4. Memory efficiency by loading data in batches
        # 5. Automatic memory pinning through pin_memory (useful for GPU training)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                        shuffle=train)


    # def get_dataloader_from_scratch(self, train):
    #     """
    #     Generate a data loader for training or validation.

    #     Args:
    #         train (bool): Whether to generate a data loader for training. If False, generate a data loader for validation.

    #     Yields:
    #         tuple: A tuple containing a batch of input data and a batch of output data.
    #     """
    #     # If it is a training data loader, generate a list of indices from 0 to num_train-1
    #     if train:
    #         indices = list(range(0, self.num_train))
    #         # The examples are read in random order
    #         random.shuffle(indices)
    #     # If it is a validation data loader, generate a list of indices from num_train to num_train+num_val-1
    #     else:
    #         indices = list(range(self.num_train, self.num_train+self.num_val))
    #     # Iterate over the indices in steps of batch_size
    #     for i in range(0, len(indices), self.batch_size):
    #         # Generate a batch of indices
    #         batch_indices = torch.tensor(indices[i: i+self.batch_size])
    #         # Yield a batch of input data and a batch of output data
    #         yield self.X[batch_indices], self.y[batch_indices]


class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)

import torchvision
from torchvision import transforms


class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
