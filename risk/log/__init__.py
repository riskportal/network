"""
risk/log
~~~~~~~~
"""

from .console import logger, log_header, set_global_verbosity
from .parameters import Params

# Initialize the global parameters logger
params = Params()
params.initialize()
