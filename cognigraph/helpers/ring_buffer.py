import numpy as np


class RingBuffer(object):

    def __init__(self, row_cnt, maxlen):
        self.maxlen = maxlen
        self.row_cnt = row_cnt
        self._data = np.empty((row_cnt, maxlen))
        self._current_sample_cnt = 0  # self._data[:, :self._filled_up_to] contains meaningful data

    def extend(self, array):
        self.check_shape(array)

        new_sample_cnt = array.shape[1]
        total_sample_cnt = self._current_sample_cnt + new_sample_cnt
        samples_to_pop_cnt = max(total_sample_cnt - self.maxlen, 0)
        samples_to_keep_cnt = self._current_sample_cnt - samples_to_pop_cnt

        self._current_sample_cnt = min(total_sample_cnt, self.maxlen)  # the number of samples after adding the new ones

        self._data = np.roll(self._data, shift=-samples_to_pop_cnt, axis=1)  # pop samples that are too old to keep
        # Write new samples. Every new sample that fits goes in
        self._data[:, samples_to_keep_cnt:self._current_sample_cnt] = array[:, -self.maxlen:]

    def check_shape(self, array):
        if array.shape[0] != self.row_cnt:
            msg = 'Wrong shape. You are trying to extend a buffer with {} rows with an array with {} rows'.format(
                self.row_cnt, array.shape[0])
            raise ValueError(msg)

    def clear(self):
        self._current_sample_cnt = 0

    @property
    def data(self):
        return self._data[:, :self._current_sample_cnt]