


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.num_bad_epochs = 0
        self.stop = False

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Loss improved from {self.best_loss:.4f} to {current_loss:.4f}. Resetting patience.")
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"Loss did not improve. Patience: {self.num_bad_epochs}/{self.patience}")
            if self.num_bad_epochs >= self.patience:
                self.stop = True
        return self.stop