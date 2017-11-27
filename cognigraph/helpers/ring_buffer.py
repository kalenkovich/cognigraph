import numpy as np


class RingBufferSlow(object):
    """Represents a multi-row deque object"""
    TIME_AXIS = 1

    def __init__(self, row_cnt, maxlen):
        self.maxlen = maxlen
        self.row_cnt = row_cnt
        self._data = np.zeros((row_cnt, maxlen))
        self._start = 0
        self._current_sample_count = 0

    def extend(self, array):
        self._check_input_shape(array)
        new_sample_cnt = array.shape[self.TIME_AXIS]

        # If new data will take all the space, we can forget about the old data
        if new_sample_cnt >= self.maxlen:
            indices = np.arange(-self.maxlen, 0) + new_sample_cnt  # last self.maxlen samples
            self._data = array.take(indices=indices, axis=1)
            self._start = 0
            self._current_sample_count = self.maxlen

        else:
            # New data should start after the end of the old one
            new_data_start = (self._start + self._current_sample_count) % self.maxlen

            # Put as much as possible after new_data_start.
            new_data_end = min(new_data_start + new_sample_cnt, self.maxlen)
            self._data[:, new_data_start:new_data_end] = array[:, :(new_data_end-new_data_start)]

            # Then wrap around if needed
            if new_data_end - new_data_start < new_sample_cnt:
                new_data_end = (new_data_start + new_sample_cnt) % self.maxlen
                self._data[:, :new_data_end] = array[:, -new_data_end:]

            self._current_sample_count = min(self._current_sample_count + new_sample_cnt, self.maxlen)
            if self._current_sample_count == self.maxlen:  # The buffer is fully populated
                self._start = new_data_end % self.maxlen

    def _check_input_shape(self, array):
        if array.shape[0] != self.row_cnt:
            msg = 'Wrong shape. You are trying to extend a buffer with {} rows with an array with {} rows'.format(
                self.row_cnt, array.shape[0])
            raise ValueError(msg)

    def clear(self):
        self._current_sample_count = 0
        self._start = 0

    @property
    def data(self):
        indices = self._start + np.arange(self._current_sample_count)
        return self._data.take(indices=indices, axis=self.TIME_AXIS, mode='wrap')

    @property
    def test_data(self):
        return np.concatenate((self._data[:, self._start:], self._data[:, :self._start]), axis=1)


class RingBuffer(object):
    """Represents a multi-row deque object.
    Very memory-inefficient (all data is saved twice). This allows us to return views and not copies of the data.
    """

    TIME_AXIS = 1

    def __init__(self, row_cnt, maxlen):
        self.maxlen = maxlen
        self.row_cnt = row_cnt
        self._data = np.zeros((row_cnt, maxlen * 2))
        self._start = 0
        self._current_sample_count = 0

    def extend(self, array):
        self._check_input_shape(array)
        new_sample_cnt = array.shape[self.TIME_AXIS]

        # If new data will take all the space, we can forget about the old data
        if new_sample_cnt >= self.maxlen:
            self._data[:, :self.maxlen] = array[:, -self.maxlen:]
            self._data[:, self.maxlen:] = array[:, -self.maxlen:]
            self._start = 0
            self._current_sample_count = self.maxlen

        else:
            # New data should start after the end of the old one
            new_data_start = (self._start + self._current_sample_count) % self.maxlen

            # Put as much as possible after new_data_start.
            new_data_end = min(new_data_start + new_sample_cnt, self.maxlen)
            self._data[:, new_data_start:new_data_end] = array[:, :(new_data_end-new_data_start)]
            self._data[:, (new_data_start + self.maxlen):(new_data_end + self.maxlen)] \
                = array[:, :(new_data_end-new_data_start)]

            # Then wrap around if needed
            if new_data_end - new_data_start < new_sample_cnt:
                new_data_end = (new_data_start + new_sample_cnt) % self.maxlen
                self._data[:, :new_data_end] = array[:, -new_data_end:]
                self._data[:, self.maxlen:(new_data_end + self.maxlen)] = array[:, -new_data_end:]

            self._current_sample_count = min(self._current_sample_count + new_sample_cnt, self.maxlen)
            if self._current_sample_count == self.maxlen:  # The buffer is fully populated
                self._start = new_data_end % self.maxlen

    def _check_input_shape(self, array):
        if array.shape[0] != self.row_cnt:
            msg = 'Wrong shape. You are trying to extend a buffer with {} rows with an array with {} rows'.format(
                self.row_cnt, array.shape[0])
            raise ValueError(msg)

    def clear(self):
        self._current_sample_count = 0
        self._start = 0

    @property
    def data(self):
        return self._data[:, self._start:(self._start + self._current_sample_count)]
