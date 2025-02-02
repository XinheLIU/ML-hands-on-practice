from d2l import torch as d2l
import torch
from torch import nn

class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        Initialize the Trainer with hyperparameters.

        Args:
            max_epochs (int): Maximum number of training epochs.
            num_gpus (int): Number of GPUs to use (currently not supported).
            gradient_clip_val (float): Value for gradient clipping.
        """
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
    
    def fit(self, model, data):
        """
        Fit the model to the data.

        Args:
            model: The model to be trained.
            data: The data to train the model on.
        """
        # Prepare data loaders and model for training
        self.prepare_data(data)  # Set up train/val data loaders
        self.prepare_model(model)  # Initialize model and visualization board
        
        # Get optimizer from model's configuration
        self.optim = model.configure_optimizers()
        
        # Initialize training state variables
        self.epoch = 0  # Current epoch number
        self.train_batch_idx = 0  # Index to track training batches
        self.val_batch_idx = 0  # Index to track validation batches
        
        # Main training loop - iterate for max_epochs
        for self.epoch in range(self.max_epochs):
            # Train and validate for one epoch
            self.fit_epoch()

    def prepare_data(self, data):
        """
        Prepare the data loaders for training and validation.

        Args:
            data: An object containing train and validation data loaders.
        """
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        """
        Prepare the model for training.

        Args:
            model: The model to be trained.
        """
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model


    def prepare_batch(self, batch):
        """
        Prepare a batch of data for training.

        Args:
            batch: A batch of data.

        Returns:
            The prepared batch.
        """
        return batch

    def fit_epoch(self):
        """
        Train the model for one epoch and validate it.
        """
        # Set model to training mode 
        self.model.train()
        # Iterate through batches in training dataloader
        for batch in self.train_dataloader:
            # Forward pass and compute loss using model's training_step
            loss = self.model.training_step(self.prepare_batch(batch))
            # Zero out gradients from previous batch
            self.optim.zero_grad()
            # Disable gradient computation for backward pass and optimization
            with torch.no_grad():
                # Compute gradients through backpropagation
                loss.backward()
                # Apply gradient clipping if specified to prevent exploding gradients
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                # Update model parameters using optimizer
                self.optim.step()
            # Increment training batch counter
            self.train_batch_idx += 1

        # Skip validation if no validation dataloader provided
        if self.val_dataloader is None:
            return
            
        # Set model to evaluation mode - disables dropout, uses running stats for batch norm
        self.model.eval()
        # Iterate through validation batches
        for batch in self.val_dataloader:
            # Disable gradient computation for validation
            with torch.no_grad():
                # Compute validation metrics using model's validation_step
                self.model.validation_step(self.prepare_batch(batch))
            # Increment validation batch counter
            self.val_batch_idx += 1

# d2l.Module
# def apply_init(self, inputs, init=None):
#     self.forward(*inputs)
#     if init is not None:
#         self.net.apply(init)

class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        """
        Initialize the LinearRegressionScratch model.

        Args:
            num_inputs (int): Number of input features.
            lr (float): Learning rate for the optimizer.
            sigma (float): Standard deviation for weight initialization.
        """
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        """
        Compute the loss for the model.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): True values.

        Returns:
            torch.Tensor: Computed loss.
        """
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.SGD: The configured optimizer.
        """
        return torch.optim.SGD([self.w, self.b], self.lr)
    
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        """
        Initialize the SGD optimizer.

        Args:
            params: Parameters to optimize.
            lr (float): Learning rate.
        """
        self.save_hyperparameters()

    def step(self):
        """
        Perform a single optimization step.
        """
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        """
        Zero the gradients of all optimized parameters.
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        """
        Initialize the LinearRegression model.

        Args:
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self, X):
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(X)
    
    def loss(self, y_hat, y):
        """
        Compute the loss for the model.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): True values.

        Returns:
            torch.Tensor: Computed loss.
        """
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.SGD: The configured optimizer.
        """
        return torch.optim.SGD(self.parameters(), self.lr)

def l2_penalty(w):
    """
    Compute the L2 penalty for regularization.

    Args:
        w (torch.Tensor): Weights of the model.

    Returns:
        torch.Tensor: L2 penalty.
    """
    return (w ** 2).sum() / 2

class WeightDecayScratch(d2l.LinearRegressionScratch):
    """Linear regression with L2 regularization implemented from scratch."""
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        """
        Initialize the WeightDecayScratch model.

        Args:
            num_inputs (int): Number of input features.
            lambd (float): L2 regularization parameter.
            lr (float): Learning rate for the optimizer.
            sigma (float): Standard deviation for weight initialization.
        """
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        """
        Compute the loss with L2 regularization.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): True values.

        Returns:
            torch.Tensor: Computed loss with L2 penalty.
        """
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
    
class WeightDecay(d2l.LinearRegression):
    """Linear regression with L2 regularization using high-level APIs."""
    def __init__(self, wd, lr):
        """
        Initialize the WeightDecay model.

        Args:
            wd (float): Weight decay (L2 regularization) parameter.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.SGD: The configured optimizer with weight decay.
        """
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)