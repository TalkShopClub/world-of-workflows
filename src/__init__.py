"""
World of Workflows - Source Package

This package contains all source code for the world model evaluation system.
"""

__version__ = "0.1.0"
__author__ = "World of Workflows Team"

# Import key modules for easier access
try:
    from .world_model_agent import WorldModelAgent, StateDiff, ActionCall
except ImportError:
    pass

