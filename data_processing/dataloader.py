import numpy as np

class DataLoader:
    def __init__(self, method="standard", batch_size=32, drop_last=False, shuffle=True):
        self.enabled = True
        self.method = method
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def get_batch(self, X, y):
        """ Dispatches data to correct batching generator"""
        if self.method == "standard":
            return self._standard_batch(X, y)
        elif self.method == "stratified":
            return self._stratified_batch(X, y)
        else:
            raise ValueError(f"Unknown batching method: {self.method}")

    def _standard_batch(self, X, y):
        m = X.shape()[0] # samples
        indices = np.random.permutation(m) if self.shuffle else np.arange(m)

        # Calculate stopping point depending on drop_last (set to True during training, False during CV & Test)
        end_idx = m - (m % self.batch_size) if self.drop_last else m

        for i in range(0, end_idx, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _stratified_batch(self, X, y):
        pass

