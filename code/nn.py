import autograd.numpy as np
from autograd.util import flatten_func
import copy


# Activations

def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1.0)


def relu(x, alpha=0.0):
    if alpha == 0.0:
        return 0.5 * (x + np.abs(x))
    else:
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * np.abs(x)


def logsumexp(X):
    max_X = np.max(X, axis=-1)[..., np.newaxis]
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=-1)[..., np.newaxis])


def softmax(X):
    return np.exp(X - logsumexp(X))


# Initializations

def glorot_uniform(fan_in, fan_out, rng, scale=0.1):
    s = scale * np.sqrt(6. / (fan_in + fan_out))
    if fan_out > 0:
        return rng.rand(fan_in, fan_out) * 2 * s - s
    else:
        return rng.rand(fan_in) * 2 * s - s


def orthogonal(shape, rng, scale=1.1):
    # from Keras
    a = rng.normal(0.0, 1.0, shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == shape else v
    q = q.reshape(shape)
    return scale * q


# Training

class AdamOptimizer:
    def __init__(self, grad, init_params, callback=None,
                 step_size=0.01, b1=0.9, b2=0.999, eps=10**-8):
        self.grad = grad
        self.init_params = copy.copy(init_params)
        self.callback = callback
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.flattened_grad, self.unflatten, self.x = flatten_func(
            self.grad, self.init_params)
        self.reset()

    def reset(self):
        self.m = np.zeros(len(self.x))
        self.v = np.zeros(len(self.x))
        self.counter = 0

    def step(self,  num_iters=1):
        for i in range(num_iters):
            g = self.flattened_grad(self.x, self.counter)
            # XXX add clipping

            if self.callback:
                self.callback(self.unflatten(self.x),
                              self.counter,
                              self.unflatten(g))

            self.m = (1 - self.b1) * g + self.b1 * self.m
            self.v = (1 - self.b2) * (g ** 2) + self.b2 * self.v
            mhat = self.m / (1 - self.b1 ** (self.counter + 1))
            vhat = self.v / (1 - self.b2 ** (self.counter + 1))
            self.x = self.x - self.step_size*mhat/(np.sqrt(vhat) + self.eps)
            self.counter += 1

        return self.unflatten(self.x)

    def move_to(self, params):
        for k, v in self.unflatten(self.x).items():
            params[k] = v
        return params
