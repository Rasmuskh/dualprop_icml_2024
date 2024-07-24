from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import grad, device_put
from jax._src import core
from jax._src import dtypes
from jax._src.typing import Array
from jax._src.nn.initializers import variance_scaling, he_uniform, he_normal, lecun_normal, lecun_uniform
from typing import Callable, Tuple, Union, List, Sequence, Any
from abc import abstractmethod, ABC
import sys
import math
sys.path.append("/src/")

class conv_mp_block(nn.Module):
    ConvLayer: Callable
    features: int
    kernel_size: Union[int, Tuple[int, int]]
    strides: Union[int, Tuple[int, int]]
    dtype: Union[jnp.float32, jnp.float16, jnp.bfloat16]
    param_dtype: Union[jnp.float32, jnp.bfloat16, jnp.float16]
    name: str
    pooling: Callable # either maxpool2x2 or identity

    def setup(self):
        self.conv = self.ConvLayer(#precision=jax.lax.Precision("highest"),
                                   features=self.features, kernel_size=self.kernel_size, 
                                   strides=self.strides, padding='same', name='conv_mp',
                                   param_dtype=self.param_dtype)
    def __call__(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        return x
    
    def call_without_pooling(self, x):
        x = self.conv(x)
        return x

def flatten(x):
    return x.reshape((x.shape[0], -1))

def identity(x):
    return x

def maxpool2x2(x):
    return nn.max_pool(x, window_shape=(2,2), strides=(2,2))

class cnn_abstract(nn.Module, ABC):
    loss: Callable
    ConvLayer: Callable
    DenseLayer: Callable
    act: Callable
    num_classes: int
    beta: float
    alpha: float
    dtype: Union[jnp.float32, jnp.bfloat16, jnp.float16]
    param_dtype: Union[jnp.float32, jnp.bfloat16, jnp.float16]
    kernels: List[Tuple[int, int]]
    strides: List[Tuple[int, int]]
    mp: List[int]
    features: List[int]
    dense_features: List[int]
    inference_sequence: str
    inference_passes_nudged: int

    def setup(self):
        layers = []
        # create conv layers
        for i in range(0, len(self.features)):
            h = maxpool2x2 if self.mp[i]==True else identity
            layers.append(conv_mp_block(self.ConvLayer, self.features[i], self.kernels[i], strides=self.strides[i], name=f'c{i:02d}', 
                                        dtype=self.dtype, param_dtype=self.param_dtype, pooling=h))
        # create first dense layer
        layers.append(nn.Sequential([flatten, self.DenseLayer(#precision=jax.lax.Precision("highest"),
                                                              features=self.dense_features[0], name='d00', 
                                                              param_dtype=self.param_dtype)]))
        # create subsequent dense layers
        for i in range(1, len(self.dense_features)):
            layers.append(self.DenseLayer(#precision=jax.lax.Precision("highest"),
                                          features=self.dense_features[i], name=f'd{i:02d}', 
                                          param_dtype=self.param_dtype))
        self.layers = layers
        self.num_convlayers = len(self.features)
        self.num_denselayers = len(self.dense_features)
        self.num_layers = self.num_convlayers + self.num_denselayers

        L = len(self.layers)
        def get_inf_seq(inference_passes):
            if self.inference_sequence == 'fwbwK':
                downagain = [i for i in range(L,0, -1)]
                upanddown = [i for i in range(1,L)] + [i for i in range(L,0, -1)]
                inf_seq = downagain[0:-1] + upanddown*(inference_passes - 1) + [1]
            elif self.inference_sequence == 'fwK':
                inf_seq = [i for i in range(1,L+1)]*inference_passes
            elif self.inference_sequence == 'bwK':
                inf_seq = [i for i in range(L,0,-1)]
            elif self.inference_sequence == 'oddeven':
                inf_seq = ([i for i in range(1, L+1, 2)] + [i for i in range(2, L+1, 2)])*inference_passes
            elif self.inference_sequence == 'evenodd':
                inf_seq = ([i for i in range(2, L+1, 2)] + [i for i in range(1, L+1, 2)])*inference_passes
            return inf_seq
        
        self.inf_seq_nudged = get_inf_seq(self.inference_passes_nudged)

    def __call__(self, x):
        """Perform a forward pass through the layers and return output neurons."""
        for layer in self.layers[0:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x

    def ff_with_hiddens(self, x0):
        """Perform a forward pass through the layers and return list of input, hidden neurons and output neurons."""
        s = [x0]
        for layer in self.layers[0:-1]:
            s.append(self.act(layer(s[-1])))
        s.append(self.layers[-1](s[-1]))
        return s
    
    def init_states_to_zero(self, x0):
        s = [x0]
        for layer in self.layers[0:-1]:
            s.append(0.0*layer(s[-1]))
        s.append(0.0*self.layers[-1](s[-1]))
        return s
    
    def make_predictions(self, x):
        return self.ff_with_hiddens(x)
    
    def output_loss(self, logits, targets):
        return self.loss(logits, targets)
    
    def get_phi(self, s_i, s_im1, layer_in):
        phi = jnp.sum(layer_in(s_im1)*s_i)
        return phi
    
    def get_phi_no_pooling(self, s_i, s_im1, layer_in):
        phi = jnp.sum(layer_in.call_without_pooling(s_im1)*s_i)
        return phi
    
    def get_L_and_gamma(self, s, rng_key, numiter):
        key_a, key_b = jax.random.split(rng_key)
        rng_keys_a = jax.random.split(key_a, len(s))
        b = [jax.random.normal(rngi, [1]+list(si.shape[1:]), dtype=self.dtype) for (si, rngi) in zip(s, rng_keys_a)]

        # We need to create some dummy states with larger size also (since we don't want to use maxpooling here)
        # These states are only used to differentiate with respect to, so their values don't actually matter.
        # However, it is convenient to use call_without_pooling to get something with the right shape
        bdummy = [b[0]]
        for i in range(0,self.num_convlayers):
            bdummy.append(self.layers[i].call_without_pooling(b[i]))
        L = []
        gammas = []

        for i in range(0, self.num_convlayers-1):
            for iteration in range(0,numiter):
                Wb = grad(self.get_phi_no_pooling, argnums=(0))(bdummy[i+1], b[i], self.layers[i]) # compute W@b, note b[i+1] acts as a dummy variable here
                WTWb = grad(self.get_phi_no_pooling, argnums=(1))(Wb, b[i], self.layers[i]) # Compute W^T @ (W @ b), note b[i] acts as a dummy variable here
                b[i] = WTWb / jnp.sqrt((WTWb**2).sum())
            Wb = grad(self.get_phi_no_pooling, argnums=(0))(bdummy[i+1], b[i], self.layers[i])
            gamma = jnp.sqrt((Wb**2).sum())
            gammas.append(gamma)
            L.append(jnp.maximum(0.0, (gamma**2 - 1.0)/2.0))

        for i in range(self.num_convlayers-1, self.num_layers):
            for iteration in range(0,numiter):
                Wb = grad(self.get_phi, argnums=(0))(b[i+1], b[i], self.layers[i]) # compute W@b, note b[i+1] acts as a dummy variable here
                WTWb = grad(self.get_phi, argnums=(1))(Wb, b[i], self.layers[i]) # Compute W^T @ (W @ b), note b[i] acts as a dummy variable here
                b[i] = WTWb / jnp.sqrt((WTWb**2).sum())
            Wb = grad(self.get_phi, argnums=(0))(b[i+1], b[i], self.layers[i])
            gamma = jnp.sqrt((Wb**2).sum())
            gammas.append(gamma)
            L.append(jnp.maximum(0.0, (gamma**2 - 1.0)/2.0))

        return L, gammas
    
class cnn_dualprop_abstract(cnn_abstract):

    def infer_states_train(self, x0, y, rng_key):
        # NOTE: we use a trick to compute ff and fb input as the grad of a type of Hopfield energy. 
        # This means we don't have to juggle with conv/convtranspose parameters and maxpooling indices manually.
        # splus[i] is used as a dummy argument for this gradient computation and using sminus[i] would give the same result,
        # Intuition: d/du(transpose(u)*A*v)=transpose(A)v regardless of the value of u.
        # Init with a forward pass
        splus =  self.ff_with_hiddens(x0)
        sminus = jax.device_put(splus) # copy by value
        pred = jax.device_put(splus[-1]) # copy by value

        for i in self.inf_seq_nudged:
            if i==len(splus)-1:
                sbar_previous = self.alpha*splus[-2] + (1-self.alpha)*sminus[-2]
                splus[-1], sminus[-1] = self.infer_outputs(pred, y, sbar_previous, self.layers[-1])
            else:
                sbar_previous = self.alpha*splus[i-1] + (1-self.alpha)*sminus[i-1]
                delta = splus[i+1] - sminus[i+1]
                splus[i], sminus[i] = self.infer_hidden(splus[i], sbar_previous, delta, self.layers[i-1], self.layers[i])
        return splus, sminus
    
    def get_J(self, splus, sminus):
        J = 0.0
        batchsize = splus[-1].shape[0]
        for i in range(1,len(splus)):
            sbar_previous = self.alpha*splus[i-1] + (1-self.alpha)*sminus[i-1]
            delta = splus[i] - sminus[i]
            J += self.get_phi(-delta, sbar_previous, self.layers[i-1])/self.beta
        return J/batchsize


    @abstractmethod
    def infer_hidden(self, s, sbar_previous, delta, layer_in, layer_out):
        pass

    @abstractmethod
    def infer_outputs(self, s, y, sbar_previous, layer_in):
        pass

class cnn_dualprop_Lagr_ff(cnn_dualprop_abstract):
    
    def infer_hidden(self, s, sbar_previous, delta, layer_in, layer_out):
        # Note splus_k is a dummy argument when computing ff and fb
        ff = grad(self.get_phi, argnums=(0))(s, sbar_previous, layer_in)
        fb = grad(self.get_phi, argnums=(1))(delta, s, layer_out)
        splus_k = self.act(ff + (1 - self.alpha) * fb)
        sminus_k = self.act(ff - self.alpha * fb)
        return splus_k, sminus_k
    
    def infer_outputs(self, pred, y, sbar_previous, layer_in):
        ff = grad(self.get_phi, argnums=(0))(pred, sbar_previous, layer_in) # pred is a dummy arg here
        fb = self.beta*grad(self.loss, argnums=(0))(pred, y)
        splus_out = ff - (1 - self.alpha) * fb
        sminus_out = ff + self.alpha * fb
        return splus_out, sminus_out

class cnn_dualprop_RAOVR_ff(cnn_dualprop_abstract):
    
    def infer_hidden(self, s, sbar_previous, delta, layer_in, layer_out):
        # Note splus_k is a dummy argument when computing ff and fb
        ff = grad(self.get_phi, argnums=(0))(s, sbar_previous, layer_in)
        fb = grad(self.get_phi, argnums=(1))(delta, s, layer_out)
        splus_k = self.act(ff + self.alpha * fb)
        sminus_k = self.act(ff - (1 - self.alpha) * fb)
        return splus_k, sminus_k

    def infer_outputs(self, s, y, sbar_previous, layer_in):
        ff = grad(self.get_phi, argnums=(0))(s, sbar_previous, layer_in)
        fb = self.beta*grad(self.loss, argnums=(0))(ff, y)
        splus_out = ff - self.alpha * fb
        sminus_out = ff + (1 - self.alpha) * fb
        return splus_out, sminus_out
    
class cnn_dualprop_RAOVR_dampened_ff(cnn_dualprop_abstract):
    
    def infer_hidden(self, splus_k, sminus_k, sbar_km1, delta, layer_in, layer_out, L):
        # Note splus_k is a dummy argument when computing ff and fb
        ff = grad(self.get_phi, argnums=(0))(splus_k, sbar_km1, layer_in)
        fb = grad(self.get_phi, argnums=(1))(delta, splus_k, layer_out)
        # for t in range(0,5):
        splus_k = self.act((ff + self.alpha * fb + L*splus_k)/(1+L))
        sminus_k = self.act((ff - (1 - self.alpha) * fb + L*sminus_k)/(1+L))
        return splus_k, sminus_k
    
    def infer_outputs(self, pred, y, sbar_previous, layer_in):
        ff = grad(self.get_phi, argnums=(0))(pred, sbar_previous, layer_in) #pred is a dummy arg here
        fb = self.beta*grad(self.loss, argnums=(0))(pred, y)
        splus_out = ff - self.alpha * fb
        sminus_out = ff + (1 - self.alpha) * fb
        return splus_out, sminus_out
    
    def infer_states_train(self, x0, y, rng_key):
        # sstar is just the activations found by make_predictions
        splus =  self.ff_with_hiddens(x0)
        sminus = jax.device_put(splus) # copy by value
        pred = jax.device_put(splus[-1]) # copy by value
        L, gamma = self.get_L_and_gamma(splus, rng_key, numiter=50)
        for k in self.inf_seq_nudged:
            if k==len(splus)-1:
                sbar_km1 = self.alpha*splus[-2] + (1-self.alpha)*sminus[-2]
                splus[-1], sminus[-1] = self.infer_outputs(pred, y, sbar_km1, self.layers[-1])
            else:
                sbar_km1 = self.alpha*splus[k-1] + (1-self.alpha)*sminus[k-1]
                delta = splus[k+1] - sminus[k+1]
                splus[k], sminus[k] = self.infer_hidden(splus[k], sminus[k], sbar_km1, delta, self.layers[k-1], self.layers[k], L[k])

        return splus, sminus
    
# class cnn_LPOM_ff(cnn_abstract):

#     def infer_hidden(self, si, si_m1, delta, layer_in, layer_out, L):
#         ff = grad(self.get_phi, argnums=(0))(si, si_m1, layer_in)
#         fb = grad(self.get_phi, argnums=(1))(delta, si, layer_out)
#         si = self.act((ff + fb + L*si)/(1+L))
#         fa_i = self.act(ff)
#         return si, fa_i

#     def infer_outputs(self, si, y, si_m1, layer_in):
#         ff = grad(self.get_phi, argnums=(0))(si, si_m1, layer_in)
#         fb = self.beta*grad(self.loss, argnums=(0))(ff, y)
#         s_L = ff - fb
#         fa_L = ff
#         return s_L, fa_L

#     def get_L(self, s, rng_key):
#         rng_keys = jax.random.split(rng_key, len(s))
#         b = [jax.random.normal(rngi, [1]+list(si.shape[1:]), dtype=self.dtype) for (si, rngi) in zip(s, rng_keys)]
#         L = []
#         for (i, layer) in enumerate(self.layers):
#             for iteration in range(0,3):
#                 # compute W@b, note b[i+1] acts as a dummy variable here
#                 Wb = grad(self.get_phi, argnums=(0))(b[i+1], b[i], layer)
#                 # Compute W^T @ (W @ b), note b[i] acts as a dummy variable here
#                 WTWb = grad(self.get_phi, argnums=(1))(Wb, b[i], layer)
#                 b[i] = WTWb / jnp.sqrt((WTWb**2).sum())
#             Wb = grad(self.get_phi, argnums=(0))(b[i+1], b[i], layer)
#             gamma = jnp.sqrt((Wb**2).sum())
#             L.append(1.0 + jnp.maximum(0.0, (gamma - 1.0)/2.0))
#         return L

#     def infer_states_train(self, sstar, y, rng_key):
#         # sstar is just the activations found by make_predictions
#         fa = jax.device_put(sstar) # copy by value: fa[k] is f(W_{k-1}s_{k-1}) 
#         s = jax.device_put(sstar) # copy by value
#         L = self.get_L(s, rng_key)
#         for i in self.inf_seq_nudged:
#             if i==len(s)-1:
#                 s[-1], fa[-1] = self.infer_outputs(s[-1], y, s[-2], self.layers[-1])
#             else:
#                 delta = s[i+1] - fa[i+1]
#                 s[i], fa[i] = self.infer_hidden(s[i], s[i-1], delta, self.layers[i-1], self.layers[i], L[i])

#         return s, fa