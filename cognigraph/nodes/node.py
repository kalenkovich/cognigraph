from typing import Tuple, Dict
from contextlib import contextmanager

import numpy as np
import mne
from mne.io.pick import channel_type

from ..helpers.misc import class_name_of


class Message(object):
    """Class to hold messages that need to be delivered to the descendant nodes before they update"""
    def __init__(self, there_has_been_a_change=False, output_history_is_no_longer_valid=False):
        self.there_has_been_a_change = there_has_been_a_change  # This is used as a message to the next node telling it
        # that either this or one of the nodes before it had a change
        self.output_history_is_no_longer_valid = there_has_been_a_change  # The change was such that new outputs cannot
        # be considered as continuation of the previous ones


class Node(object):
    """ Any processing step (including getting and outputting data) is an instance of this class.
    This is an abstract class.
    """

    @property
    def CHANGES_IN_THESE_REQUIRE_RESET(self) -> Tuple[str]:
        """A constant tuple of attributes after a change in which a reset should be scheduled."""
        msg = 'Each subclass of Node must have a CHANGES_IN_THESE_REQUIRE_RESET constant defined'
        raise NotImplementedError(msg)

    @property
    def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self) -> Tuple[str]:
        """A constant tuple of attributes after an *upstream* change in which an initialization should be scheduled."""
        msg = 'Each subclass of Node must have a CHANGES_IN_THESE_REQUIRE_REINITIALIZATION constant defined'
        raise NotImplementedError(msg)

    def __init__(self):
        self._input_node = None  # type: Node
        self.output = None  # type: np.ndarray
        self._receivers = {}  # type: Dict[Node, object]  # Nodes that have set this node as their input_node.

        self._initialized = False

        # Set by the input node
        self._there_has_been_an_upstream_change = False

        # Flags for different kinds of possibly needed resets
        self._should_reinitialize = False
        self._should_reset = False
        self._input_history_is_no_longer_valid = False

        self._saved_from_upstream = None  # type: dict  # Used to determine whether upstream changes warrant
        # reinitialization

    @property
    def input_node(self):
        return self._input_node

    @input_node.setter
    def input_node(self, value):
        if self._input_node is value:  # Also covers the case when both are None
            return

        # Reinitialize if has been initialized
        self._should_reinitialize = self._initialized

        # Tell the previous input node about disconnection
        if self._input_node is not None:
            self._input_node.deregister_a_receiver(self)

        self._input_node = value

        # Tell the new input node about the connection
        if value is not None:
            value.register_a_receiver(self)

    def register_a_receiver(self, receiver_node):
        self._receivers[receiver_node] = None
        # New input node means everything has changed
        receiver_node.receive_a_message(Message(
            there_has_been_a_change=True,
            output_history_is_no_longer_valid=True
        ))

    def receive_a_message(self, message: Message):
        self._there_has_been_an_upstream_change = message.there_has_been_a_change
        self._input_history_is_no_longer_valid = message.output_history_is_no_longer_valid

    def deregister_a_receiver(self, receiver_node):
        self._receivers.pop(receiver_node, None)

    def initialize(self):
        if self._initialized is True and self._should_reinitialize is False:
            raise ValueError('Trying to initialize even though there is no indication for it.')

        self._saved_from_upstream = {item: self.traverse_back_and_find(item) for item
                                     in self.UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION}

        with self.not_triggering_reset():
            print('Initializing the {} node'.format(class_name_of(self)))
            self._initialize()
            self._initialized = True

            self._no_pending_changes = True  # Set all the resetting flags to false
            self._there_has_been_an_upstream_change = False

            # Tell the receivers about what has happened
            message = Message(there_has_been_a_change=True,
                              output_history_is_no_longer_valid=True)
            self._deliver_a_message_to_receivers(message)


    def _initialize(self):
        """Prepares everything for the first update. If called again, should remove all the traces from the past"""
        raise NotImplementedError

    @property
    def _no_pending_changes(self):
        """Checks if there is any kind of reset scheduled"""
        return (self._should_reinitialize is False
                and self._input_history_is_no_longer_valid is False
                and self._should_reset is False)

    @ _no_pending_changes.setter
    def _no_pending_changes(self, value):
        if value is True:
            self._should_reinitialize = False
            self._input_history_is_no_longer_valid = False
            self._should_reset = False
        else:
            raise ValueError('Can only be set to True')

    def _deliver_a_message_to_receivers(self, message: Message):
        for receiver_node in self._receivers:
            receiver_node.receive_a_message(message)

    def update(self) -> None:
        self.output = None  # Reset output in case update does not succeed

        if self._there_has_been_an_upstream_change is True:
            self._should_reinitialize = self._the_change_requires_reinitialization()
            self._there_has_been_an_upstream_change = False

        if self._initialized is True and self._no_pending_changes is True:
            self._update()
            # Discard input  # This does not work when there are multiple descendants  # TODO: come up with a way
            # if self.input_node is not None:
            #     self.input_node.output = None

        elif self._initialized is False or self._should_reinitialize is True:
            self.initialize()
        else:
            if self._should_reset is True:
                self.reset()
            if self._input_history_is_no_longer_valid is True:
                self.on_input_history_invalidation()

    def _update(self):
        raise NotImplementedError

    def reset(self):
        if self._should_reset is False:
            raise ValueError('Trying to reset even though there is no indication for it.')

        with self.not_triggering_reset():
            print('Resetting the {} node because of attribute changes'.format(class_name_of(self)))
            output_history_is_no_longer_valid = self._reset()
            self._should_reset = False

            # Tell the receivers about what has happened
            message = Message(there_has_been_a_change=True,
                              output_history_is_no_longer_valid=output_history_is_no_longer_valid)
            self._deliver_a_message_to_receivers(message)

    def _reset(self) -> bool:
        """Does what needs to be done when one of the self.CHANGES_IN_THESE_REQUIRE_RESET has been changed
        Must return whether output history is no longer valid. True if descendants should forget about anything that
        has happened before, False if changes are strictly local."""
        raise NotImplementedError

    def on_input_history_invalidation(self):
        if self._input_history_is_no_longer_valid is False:
            raise ValueError('Trying to flush history even though there is no indication for it.')

        with self.not_triggering_reset():
            print('Resetting the {} node because history is no longer valid'.format(class_name_of(self)))
            self._on_input_history_invalidation()
            self._input_history_is_no_longer_valid = False

            # Tell the receivers about what has happened
            message = Message(there_has_been_a_change=True,
                              output_history_is_no_longer_valid=True)
            self._deliver_a_message_to_receivers(message)

    def _on_input_history_invalidation(self):
        """If the node state is dependent on previous inputs, reset whatever relies on them."""
        raise NotImplementedError
        
    def traverse_back_and_find(self, item: str):
        """ This function will walk up the node tree until it finds a node with an attribute <item> """
        try:
            return getattr(self.input_node, item)
        except AttributeError:
            try:
                return self.input_node.traverse_back_and_find(item)
            except AttributeError:
                msg = 'None of the predecessors of a {} node contains attribute {}'.format(
                    class_name_of(self), item)
                raise AttributeError(msg)

    # Schedule resetting if the change in the attribute being set warrants it
    def __setattr__(self, key, value):
        self._check_value(key, value)
        if key in self.CHANGES_IN_THESE_REQUIRE_RESET:
            super().__setattr__('_should_reset', True)
            super().__setattr__('there_has_been_a_change', True)
        super().__setattr__(key, value)

    @contextmanager
    def not_triggering_reset(self):
        """Change of attributes CHANGES_IN_THESE_REQUIRE_RESET should trigger reset() but not from within the class.
        Use this context manager to suspend reset() triggering."""
        backup, self.CHANGES_IN_THESE_REQUIRE_RESET = self.CHANGES_IN_THESE_REQUIRE_RESET, ()
        try:
            yield
        finally:
            self.CHANGES_IN_THESE_REQUIRE_RESET = backup

    def _check_value(self, key, value):
        raise NotImplementedError

    def _the_change_requires_reinitialization(self):
        """Checks if anything important changed upstream wrt value captured at initialization"""
        for item, value in self._saved_from_upstream.items():
            if value != self.traverse_back_and_find(item):
                return True
        return False


class SourceNode(Node):
    """ Objects of this class read data from a source """

    # There is no 'upstream' for the sources
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()

    def __init__(self):
        super().__init__()
        self.mne_info = None  # type: mne.Info

    def initialize(self):
        self.mne_info = None
        super().initialize()
        self._check_mne_info()

    def _check_mne_info(self):
        class_name = class_name_of(self)
        error_hint = ' Check the initialize() method'

        if self.mne_info is None:
            raise ValueError('{} node has empty mne_info attribute.'.format(class_name) + error_hint)

        channel_count = len(self.mne_info['chs'])
        if len(self.mne_info['chs']) == 0:
            raise ValueError('{} node has 0 channels in its mne_info attribute.' + error_hint)

        channel_types = {channel_type(self.mne_info, i) for i in np.arange(channel_count)}
        required_channel_types = {'grad', 'mag', 'eeg'}
        if len(channel_types.intersection(required_channel_types)) == 0:
            raise ValueError('{}')

        try:
            self.mne_info._check_consistency()
        except RuntimeError as e:
            exception_message = 'The mne_info attribute of {} node is not self-consistent'.format(class_name_of(self))
            raise Exception(exception_message) from e


    def _reset(self):
        # There is nothing to reset really. So we wil just go ahead and initialize
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        pass


class ProcessorNode(Node):
    """Still an abstract class. Initially existed for clarity of inheritance only. Now handles empty inputs."""
    def __init__(self):
        self.disabled = False
        super().__init__()

    def update(self):
        if self.disabled is True:
            self.output = self.input_node.output
            return
        if self.input_node.output is None or self.input_node.output.size == 0:
            self.output = None
            return
        else:
            super().update()


class OutputNode(Node):
    """Still an abstract class. Initially existed for clarity of inheritance only. Now handles empty inputs."""
    def update(self):
        if self.input_node.output is None or self.input_node.output.size == 0:
            return
        else:
            super().update()
