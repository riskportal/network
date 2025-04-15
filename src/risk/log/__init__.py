"""
risk/log
~~~~~~~~
"""

from risk.log.console import log_header, logger, set_global_verbosity
from risk.log.parameters import Params

# Initialize the global parameters logger
params = Params()
params.initialize()
