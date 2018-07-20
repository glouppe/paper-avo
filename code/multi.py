import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt

from carl.distributions import Join
from carl.distributions import Mixture
from carl.distributions import Normal
from carl.distributions import Exponential
from carl.distributions import LinearTransform
import theano

from nn import glorot_uniform
from nn import relu
from nn import RmsPropOptimizer

from proposals import make_gaussian_proposal
from proposals import gaussian_draw
from proposals import grad_gaussian_logpdf
from proposals import grad_gaussian_entropy

from sklearn.utils import check_random_state
from scipy.spatial.distance import mahalanobis


# Global params

seed = 777
rng = check_random_state(seed)

batch_size = 64
n_epochs = 500+1
lambda_reg = 20.
gamma = 0.005

true_theta = np.array([1.0, -1.0])
make_plots = True


# Simulator

A = theano.shared(true_theta[0], name="A")
B = theano.shared(true_theta[1], name="B")
R = np.array([
    [1.31229955,  0.10499961,  0.48310515, -0.3249938,  -0.26387927],
    [0.10499961,  1.15833058, -0.55865473,  0.25275522, -0.39790775],
    [0.48310515, -0.55865473,  2.25874579, -0.52087938, -0.39271231],
    [0.3249938,   0.25275522, -0.52087938,  1.4034925,  -0.63521059],
    [-0.26387927, -0.39790775, -0.39271231, -0.63521059,  1.]])

p0 = LinearTransform(Join(components=[
                    Normal(mu=A, sigma=1),
                    Normal(mu=B, sigma=3),
                    Mixture(components=[Normal(mu=-2, sigma=1),
                                        Normal(mu=2, sigma=0.5)]),
                    Exponential(inverse_scale=3.0),
                    Exponential(inverse_scale=0.5)]), R)


def simulator(theta, n_samples, random_state=None):
    A.set_value(theta[0])
    B.set_value(theta[1])
    return p0.rvs(n_samples, random_state=random_state)

X_obs = simulator(true_theta, 20000, random_state=rng)
n_params = len(true_theta)
n_features = X_obs.shape[1]


# Proposal distribution

params_proposal = make_gaussian_proposal(n_params)


# Critic

def make_critic(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)
    params = {"W": [glorot_uniform(n_hidden, n_features, rng),
                    glorot_uniform(n_hidden, n_hidden, rng),
                    glorot_uniform(n_hidden, 0, rng, scale=0.01)],
              "b": [np.zeros(n_hidden),
                    np.zeros(n_hidden),
                    np.zeros(1)]}
    return params

params_critic = make_critic(n_features, 10, random_state=rng)


def predict(X, params):
    h = X
    h = relu(np.dot(params["W"][0], h.T).T + params["b"][0], alpha=0.2)
    h = relu(np.dot(params["W"][1], h.T).T + params["b"][1], alpha=0.2)
    h = np.dot(params["W"][2], h.T).T + params["b"][2]
    return h

grad_predict_critic = ag.elementwise_grad(predict)


def loss_critic(params_critic, i, lambda_reg=lambda_reg, batch_size=batch_size):
    y_critic = np.zeros(batch_size)
    # y_critic[:batch_size // 2] = 0.0  # 0 == fake
    y_critic[batch_size // 2:] = 1.0

    rng = check_random_state(i)

    # WGAN loss
    thetas = gaussian_draw(params_proposal, batch_size // 2, random_state=rng)
    _X_gen = np.zeros((batch_size // 2, n_features))
    for j, theta in enumerate(thetas):
        _X_gen[j, :] = simulator(theta, 1, random_state=rng).ravel()

    indices = rng.permutation(len(X_obs))
    _X_obs = X_obs[indices[:batch_size // 2]]
    X = np.vstack([_X_gen, _X_obs])

    y_pred = predict(X, params_critic)
    l_wgan = np.mean(-y_critic * y_pred + (1. - y_critic) * y_pred)

    # # Gradient penalty
    # eps = rng.rand(batch_size // 2, 1)
    # _X_hat = eps * _X_obs + (1. - eps) * _X_gen
    # grad_Dx = grad_predict_critic(_X_hat, params_critic)
    # norms = np.sum(grad_Dx ** 2, axis=1) ** 0.5
    # l_gp = np.mean((norms - 1.0) ** 2.0)

    # Stable GAN penalty 'real'
    grad_Dx = grad_predict_critic(_X_obs, params_critic)
    norms = np.sum(grad_Dx ** 2, axis=1)
    l_reg = np.mean(norms)

    return l_wgan + lambda_reg * l_reg

grad_loss_critic = ag.grad(loss_critic)


# grad_psi E_theta~q_psi, z~p_z(theta) [ d(g(z, theta) ]

def approx_grad_u(params_proposal, i, gamma=gamma):
    rng = check_random_state(i)
    grad_u = {k: np.zeros(len(params_proposal[k])) for k in params_proposal}
    grad_ent = {k: np.zeros(len(params_proposal[k])) for k in params_proposal}
    thetas = gaussian_draw(params_proposal, batch_size, random_state=rng)

    for theta in thetas:
        x = simulator(theta, 1, random_state=rng)
        dx = predict(x, params_critic).ravel()

        grad_q = grad_gaussian_logpdf(params_proposal, theta)
        for k, v in grad_q.items():
            grad_u[k] += -dx * v

    grad_entropy = grad_gaussian_entropy(params_proposal)
    for k, v in grad_entropy.items():
        grad_ent[k] += v

    M = len(thetas)

    for k in grad_u:
        grad_u[k] = 1. / M * grad_u[k] + gamma * grad_ent[k]

    return grad_u


# Training loop

opt_critic = RmsPropOptimizer(grad_loss_critic, params_critic,
                           step_size=10e-3)
opt_proposal = RmsPropOptimizer(approx_grad_u, params_proposal,
                             step_size=10e-3)

# print(predict(X_obs, params_critic).mean())
# opt_critic.step(1000)
# opt_critic.move_to(params_critic)
# print(predict(X_obs, params_critic).mean())

for i in range(n_epochs):
    print(i, params_proposal)

    # fit simulator
    opt_proposal.step(1)
    opt_proposal.move_to(params_proposal)

    # fit critic
    opt_critic.step(1)
    opt_critic.move_to(params_critic)

    # if i % 10 == 0:
    #     # # reset critic
    #     # params_critic = make_critic(n_features, 10, random_state=i)
    #     # opt_critic = AdamOptimizer(grad_loss_critic, params_critic,
    #     #                            step_size=10e-4, b1=0.5, b2=0.9)
    #     # opt_critic.step(1000)
    #     # opt_critic.move_to(params_critic)
    #
    #     # log
    #     print(predict(X_obs, params_critic).mean())
    #
    #     thetas = gaussian_draw(params_proposal,
    #                            5000, random_state=i)
    #     _X_gen = np.zeros((5000, n_features))
    #     for j, theta in enumerate(thetas):
    #         _X_gen[j, :] = simulator(theta, 1, random_state=j).ravel()
    #     print(predict(_X_gen, params_critic).mean())

    # plot
    if make_plots:
        if i == 0:
            fig = plt.figure(figsize=(6, 3))
            ax1 = fig.add_subplot(121)

            delta = 0.025
            x = np.arange(0.0, 2.0, delta)
            y = np.arange(-2.0, 0.0, delta)
            X, Y = np.meshgrid(x, y)

            mu = params_proposal["mu"]
            sigma = np.diag(np.exp(params_proposal["log_sigma"])) ** 2.0
            sigma_inv = np.linalg.inv(sigma)

            Z = [mahalanobis(theta, mu, sigma_inv)
                 for theta in np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])]
            Z = np.array(Z).reshape(X.shape)

            CS = plt.contour(X, Y, Z, [1.0, 2.0, 3.0], colors="C1")
            fmt = {l:s for l, s in zip(CS.levels, [r"$1$", r"$2$", r"$3$"])}
            plt.clabel(CS, fmt=fmt)

            plt.scatter(mu[0], mu[1], c="C1", marker="+")
            plt.scatter(true_theta[0], true_theta[1],
                        c="C0")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")
            plt.yticks([-1.5, -1.0, -0.5])
            plt.xticks([0.5, 1.0, 1.5])

        elif i == n_epochs - 1:
            ax2 = fig.add_subplot(122)

            delta = 0.025
            x = np.arange(0.0, 2.0, delta)
            y = np.arange(-2.0, 0.0, delta)
            X, Y = np.meshgrid(x, y)

            mu = params_proposal["mu"]
            sigma = np.diag(np.exp(params_proposal["log_sigma"])) ** 2.0
            sigma_inv = np.linalg.inv(sigma)

            Z = [mahalanobis(theta, mu, sigma_inv)
                 for theta in np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])]
            Z = np.array(Z).reshape(X.shape)

            CS = plt.contour(X, Y, Z, [1.0, 2.0, 3.0], colors="C1")
            fmt = {l:s for l, s in zip(CS.levels, [r"$1$", r"$2$", r"$3$"])}
            plt.clabel(CS, fmt=fmt)

            plt.scatter(mu[0], mu[1], c="C1", marker="+")
            plt.scatter(true_theta[0], true_theta[1],
                        c="C0",
                        label=r"$\theta^* = (%d, %d)$" % (true_theta[0],
                                                          true_theta[1]))
            plt.legend(loc="upper right")
            plt.xlabel(r"$\alpha$")
            plt.ylabel(r"$\beta$")
            plt.yticks([-1.5, -1.0, -0.5])
            plt.xticks([0.5, 1.0, 1.5])

            plt.tight_layout()

            if i == n_epochs - 1:
                plt.savefig("multi-%d.pdf" % seed)

            plt.close()
