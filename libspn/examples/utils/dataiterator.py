import numpy as np
import tqdm


class DataIterator:

    def __init__(self, data, batch_size, shuffle=True):
        self._data = [data] if not isinstance(data, (list, tuple)) else data
        self._num_samples = len(data[0])
        self._ind = 0
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._pbar = None
        self._reset()
        self._increment = batch_size
        self._step = 0

    def next_batch(self, batch_size=None):
        batch_size = batch_size or self._batch_size
        end_ind = min(self._ind + batch_size, self._num_samples)

        out = tuple(d[self._ind:end_ind] for d in self._data)

        self._step += 1
        if end_ind == self._num_samples:
            self._reset()
        else:
            self._ind += batch_size
        return out

    def iter_epoch(self, desc="Epoch", batch_size=None):
        self._pbar = tqdm.trange(self.num_batches(batch_size=batch_size), desc=desc)
        for _ in self._pbar:
            yield self.augment(self.next_batch(batch_size=batch_size))
        self._pbar.close()
        self._pbar = None
        self._reset()

    def _reset(self):
        self._ind = 0
        self._step = 0
        self._permute()

    def display_progress(self, **kwargs):
        if self._pbar is None:
            raise ValueError("No progress bar to use")
        self._pbar.set_postfix(**kwargs)

    def _permute(self):
        if not self._shuffle:
            return
        perm = np.random.permutation(self._num_samples)
        self._data = [d[perm] for d in self._data]

    @property
    def end_of_epoch(self):
        return self._ind == 0

    def num_batches(self, batch_size=None):
        if batch_size is not None:
            return int(np.ceil(self._num_samples / batch_size))
        return int(np.ceil(self._num_samples / self._batch_size))

    @property
    def step(self):
        return self._step - 1

    def augment(self, minibatch):
        return minibatch