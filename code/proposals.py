import autograd as ag
import autograd.numpy as np
from autograd.scipy.special import digamma
from autograd.scipy.special import gammaln

from scipy.stats import norm as sp_norm
from scipy.stats import beta as sp_beta


# Gaussian proposal (for unbounded support)

def make_gaussian_proposal(n_parameters, mu=0.0, log_sigma=0.0):
    return {"mu":  mu * np.ones(n_parameters),
            "log_sigma": log_sigma * np.ones(n_parameters)}


def gaussian_draw(params, n_samples, random_state=None):
    n_parameters = len(params["mu"])
    mu = params["mu"]
    sigma = np.exp(params["log_sigma"])

    thetas = np.zeros((n_samples, n_parameters))

    for i in range(n_parameters):
        thetas[:, i] = sp_norm.rvs(size=n_samples,
                                   loc=mu[i],
                                   scale=sigma[i],
                                   random_state=random_state)

    return thetas


def gaussian_logpdf(params, theta, to_scalar=True):
    mu = params["mu"]
    sigma = np.exp(params["log_sigma"])

    logp = -(np.log(sigma) +
             np.log((2. * np.pi) ** 0.5) +
             (theta - mu) ** 2 / (2. * sigma ** 2))

    if to_scalar:
        return np.sum(logp)
    else:
        return logp


def gaussian_entropy(params):
    sigma = np.exp(params["log_sigma"])
    return np.sum(np.log(sigma * (2. * np.pi * np.e) ** 0.5))


grad_gaussian_logpdf = ag.grad(gaussian_logpdf)
grad_gaussian_entropy = ag.grad(gaussian_entropy)


# Truncated gaussian  (for continuous in [0,1])

# XXX


# Beta proposal (for continuous in [0,1])

def make_beta_proposal(n_parameters, log_alpha=0.0, log_beta=0.0):
    return {"log_alpha":  log_alpha * np.ones(n_parameters),
            "log_beta": log_beta * np.ones(n_parameters)}


def beta_draw(params, n_samples, random_state=None):
    n_parameters = len(params["log_alpha"])
    alpha = np.exp(params["log_alpha"])
    beta = np.exp(params["log_beta"])

    thetas = np.zeros((n_samples, n_parameters))

    for i in range(n_parameters):
        thetas[:, i] = sp_beta.rvs(size=n_samples,
                                   a=alpha[i], b=beta[i],
                                   random_state=random_state)

    return thetas


def beta_logpdf(params, theta, to_scalar=True):
    alpha = np.exp(params["log_alpha"])
    beta = np.exp(params["log_beta"])

    logp = (gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
            (alpha - 1) * np.log(theta) +
            (beta - 1) * np.log(1 - theta))

    if to_scalar:
        return np.sum(logp)
    else:
        return logp


def betaln(alpha, beta):
    return gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)


def beta_entropy(params):
    alpha = np.exp(params["log_alpha"])
    beta = np.exp(params["log_beta"])

    return np.sum(betaln(alpha, beta) -
                  (alpha - 1.0) * (digamma(alpha) - digamma(alpha + beta)) -
                  (beta - 1.0) * (digamma(beta) - digamma(alpha + beta)))


grad_beta_logpdf = ag.grad(beta_logpdf)
grad_beta_entropy = ag.grad(beta_entropy)
