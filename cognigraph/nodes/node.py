from typing import Tuple, Dict

import numpy as np

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

        self._receivers = {}  # type: Dict[Message]

        self._should_initialize = True
        self._should_reset = False
        self._saved_from_upstream = None  # type: dict  # Used to determine whether upstream changes warrant
        # reinitialization

    @property
    def input_node(self):
        return self._input_node

    @input_node.setter
    def input_node(self, value):
        if self._input_node is value:  # Also covers the case when both are None
            return

        # Tell the previous input node about disconnection
        if self._input_node is not None:
            self._input_node.deregister_a_receiver(self)

        self._input_node = value

        # Tell the new input node about the connection
        if value is not None:
            value.register_a_receiver(self)

    def register_a_receiver(self, receiver_node):
        # New input node mean everything has changed
        self._receivers[receiver_node] = Message(there_has_been_a_change=True,
                                                 output_history_is_no_longer_valid=True)

    def deregister_a_receiver(self, receiver_node):
        self._receivers.pop(receiver_node, None)

    def initialize(self):
        self._saved_from_upstream = {item: self.traverse_back_and_find(item) for item
                                     in self.UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION}

        # Attribute setting inside _initialize should not result in reset
        old, self.CHANGES_IN_THESE_REQUIRE_RESET = self.CHANGES_IN_THESE_REQUIRE_RESET, ()
        self._initialize()
        self._should_initialize = False
        self.CHANGES_IN_THESE_REQUIRE_RESET = old

    def _initialize(self):
        raise NotImplementedError

    def __setattr__(self, key, value):
        self._check_value(key, value)
        if key in self.CHANGES_IN_THESE_REQUIRE_RESET:
            super().__setattr__('_should_reset', True)
            super().__setattr__('there_has_been_a_change', True)  # This is a message for the next node
        super().__setattr__(key, value)

    def _check_value(self, key, value):
        raise NotImplementedError

    def update(self) -> None:
        self.output = None  # Reset output in case update does not succeed

        if self._there_was_a_change_upstream():
            self._react_to_the_change_upstream()  # Schedule reset or initialize
        self._reset_or_reinitialize_if_needed()  # Needed - because of a possible change in this node or upstream

        self._update()

        # Discard input  # This does not work when there are multiple descendants  # TODO: come up with a way
        # if self.input_node is not None:
        #     self.input_node.output = None

    def _update(self):
        raise NotImplementedError

    def reset(self):
        # Attribute setting inside _initialize should not result in reset
        old, self.CHANGES_IN_THESE_REQUIRE_RESET = self.CHANGES_IN_THESE_REQUIRE_RESET, ()
        self._initialize()
        self._should_reset = False
        self.CHANGES_IN_THESE_REQUIRE_RESET = old

    def _reset(self):
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

    def _there_was_a_change_upstream(self):
        """Asks the immediate predecessor node if there has a been a change in or before it"""
        if self.input_node is not None and self.input_node.there_has_been_a_change is True:
            return True
        else:
            return False

    def _react_to_the_change_upstream(self):
        """Schedules reset or reinitialization of the node depending on what has changed."""
        if self._the_change_requires_reinitialization():
            self._should_initialize = True
        else:
            self._should_reset = True

        self.input_node.there_has_been_a_change = False  # We got the message, no need to keep telling us.
        self.there_has_been_a_change = True  # We should however leave a message to the node after us.

    def _reset_or_reinitialize_if_needed(self):
        """This function is called before each _update call"""
        if self._should_initialize is True:
            self.initialize()
        elif self._should_reset is True:
            self._reset()
        self._should_initialize = False
        self._should_reset = False

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

    def _reset(self):
        # There is nothing to reset really. So we wil just go ahead and initialize
        self.initialize()


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
