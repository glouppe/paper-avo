import autograd as ag
import autograd.numpy as np
from scipy.stats import norm


def make_proposal(n_parameters):
    mu = np.zeros(n_parameters)
    beta = np.zeros(n_parameters)
    return {"mu": mu, "beta": beta}


def proposal_draw(params, n_samples, random_state=None):
    n_parameters = len(params["mu"])
    mu = params["mu"]
    sigma = np.exp(params["beta"])

    thetas = np.zeros((n_samples, n_parameters))

    for i in range(n_parameters):
        thetas[:, i] = norm.rvs(size=n_samples,
                                loc=mu[i],
                                scale=sigma[i],
                                random_state=random_state)

    return thetas


def proposal_logpdf(params, theta, eps=10e-8):
    mu = params["mu"]
    sigma = np.exp(params["beta"])

    logp = -(np.log(sigma) +
             np.log((2. * np.pi) ** 0.5) +
             (theta - mu) ** 2 / (2. * sigma ** 2))

    return np.sum(logp)


def proposal_entropy(params, theta):
    logp = proposal_logpdf(params, theta)
    return -np.exp(logp) * logp


grad_proposal_logpdf = ag.grad(proposal_logpdf)
grad_proposal_entropy = ag.grad(proposal_entropy)
