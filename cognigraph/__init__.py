__all__ = ["node", "pipeline"]

import numpy as np
# from .node import Node, SourceNode, OutputNode, ProcessorNode
# from .pipeline import Pipeline

# TODO: I wish this was an empty file

TIME_AXIS = 1
CHANNEL_AXIS = 1 - TIME_AXIS
PYNFB_TIME_AXIS = 0

MISC_CHANNEL_TYPE = 'misc'

DTYPE = np.dtype('float32')