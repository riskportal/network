"""
risk/_log
~~~~~~~~~
"""

from ._console import log_header, logger, set_global_verbosity
from ._parameters import Params

# Initialize the global parameters logger
params = Params()
params.initialize()
