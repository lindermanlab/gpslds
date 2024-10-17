import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abc import ABC, abstractmethod

class CVISitesSSM():
    """
    Adapted from Verma 2024
    Class for a discrete-time SSM
    """
    pass


class CVISitesSDE():
    """
    Adapted from Verma 2024
    Class for a continuous-time SSM i.e. an SDE
    """
    
    def __init__(self):


    