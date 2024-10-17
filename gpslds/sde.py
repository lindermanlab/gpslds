from abc import ABC, abstractmethod
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

class SDE(ABC):
    """
    From Verma 2024, an abstract base class representing a stochastic differential equation
    """

    def __init__(self, state_dim):
        super().__init__()
        self._state_dim = state_dim

    @property
    def state_dim(self) -> int:
        """
        Return the state dimension of the sde.
        """
        return self._state_dim
    
    @abstractmethod
    def drift(self, x: jnp.array, t: jnp.array) -> jnp.array:
        """
        Drift function of the SDE i.e. `f(x(t),t)`
        """
        raise NotImplementedError
    
    @abstractmethod
    def diffusion(self, x: jnp.array, t: jnp.array) -> jnp.array:
        """
        Diffusion function of the SDE i.e. `l(x(t),t)`
        """
        raise NotImplementedError
    
class OrnsteinUhlenbeckSDE(SDE, ABC):
    """
    From Verma 2024, an Ornstein-Uhlenbeck SDE represented by
    dx(t) = -λ x(t) dt + dB(t), 
    the spectral density of the Brownian motion is specified by q.
    """

    def __init__(self, decay: float = 1., q: jnp.array = jnp.ones((1, 1)), trainable: bool = False):
        """
        Initialize the Ornstein-Uhlenbeck SDE.
        """
        super(OrnsteinUhlenbeckSDE, self).__init__(state_dim=q.shape[0])
        self.q = q
        # TODO: the decay rate is not trainable 
        self.decay = decay

    def drift(self, x: jnp.array, t: jnp.array = None) -> jnp.array:
        """
        Drift of the Ornstein-Uhlenbeck process
        """
        assert x.shape[-1] == self.state_dim
        return -self.decay * x

    def diffusion(self, x: jnp.array, t: jnp.array = None) -> jnp.array:
        """
        Diffusion of the Ornstein-Uhlenbeck process
        """
        assert x.shape[-1] == self.state_dim
        return jnp.linalg.cholesky(self.q)

def linearize_sde(sde: SDE):
    pass