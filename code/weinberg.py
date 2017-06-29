import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt
import copy

from nn import glorot_uniform
from nn import relu
from nn import AdamOptimizer

# from proposals import make_gaussian_proposal
# from proposals import gaussian_draw
# from proposals import gaussian_logpdf
# from proposals import grad_gaussian_logpdf
# from proposals import grad_gaussian_entropy

# XXX switch to a gaussian proposal

from proposals import make_beta_proposal
from proposals import beta_draw
from proposals import beta_logpdf
from proposals import grad_beta_logpdf
from proposals import grad_beta_entropy

from sklearn.utils import check_random_state

# Global params

batch_size = 64
n_epochs = 300+1
lambda_gp = 0.025
gamma = 0.0

true_theta = np.array([(42 - 40) / (50 - 40)])
make_plots = True


# Simulator

def a_fb(sqrtshalf, gf):
    MZ = 90
    GFNom = 1.0
    sqrts = sqrtshalf * 2.
    A_FB_EN = np.tanh((sqrts - MZ) / MZ * 10)
    A_FB_GF = gf / GFNom

    return 2 * A_FB_EN*A_FB_GF


def diffxsec(costheta, sqrtshalf, gf):
    norm = 2. * (1. + 1. / 3.)
    return ((1 + costheta ** 2) + a_fb(sqrtshalf, gf) * costheta) / norm

def rej_sample_costheta(n_samples, theta, rng):
    sqrtshalf = theta[0]
    gf = 0.9

    ntrials = 0
    samples = []
    x = np.linspace(-1, 1, num=1000)
    maxval = np.max(diffxsec(x, sqrtshalf, gf))

    while len(samples) < n_samples:
        ntrials = ntrials+1
        xprop = rng.uniform(-1, 1)
        ycut = rng.rand()
        yprop = diffxsec(xprop, sqrtshalf, gf)/maxval
        if yprop/maxval < ycut:
            continue
        samples.append(xprop)

    return np.array(samples)

def simulator(theta, n_samples, random_state=None):
    theta = copy.copy(theta)
    theta[0] = theta[0] * (50 - 40) + 40

    rng = check_random_state(random_state)
    samples = rej_sample_costheta(n_samples, theta, rng)

    return samples.reshape(-1, 1)

X_obs = simulator(true_theta, 20000, random_state=123)
n_params = len(true_theta)
n_features = X_obs.shape[1]


# Proposal distribution

params_proposal = make_beta_proposal(n_params,
                                     log_alpha=np.log(2.0),
                                     log_beta=np.log(2.0))


# Critic

def make_critic(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)
    params = {"W": [glorot_uniform(n_hidden, 1, rng),
                    glorot_uniform(n_hidden, n_hidden, rng),
                    glorot_uniform(n_hidden, n_hidden, rng),
                    glorot_uniform(n_hidden, 0, rng)],
              "b": [np.zeros(n_hidden),
                    np.zeros(n_hidden),
                    np.zeros(n_hidden),
                    np.zeros(1)]}
    return params

params_critic = make_critic(1, 10)


def predict(X, params):
    h = X
    h = relu(np.dot(params["W"][0], h.T).T + params["b"][0], alpha=0.1)
    h = relu(np.dot(params["W"][1], h.T).T + params["b"][1], alpha=0.1)
    h = relu(np.dot(params["W"][2], h.T).T + params["b"][2], alpha=0.1)
    h = np.dot(params["W"][3], h.T).T + params["b"][3]
    return h

grad_predict_critic = ag.elementwise_grad(predict)


def loss_critic(params_critic, i, lambda_gp=lambda_gp, batch_size=batch_size):
    y_critic = np.zeros(batch_size)
    # y_critic[:batch_size // 2] = 0.0  # 0 == fake
    y_critic[batch_size // 2:] = 1.0

    rng = check_random_state(i)

    # WGAN loss
    thetas = beta_draw(params_proposal, batch_size // 2)
    _X_gen = np.zeros((batch_size // 2, n_features))
    for j, theta in enumerate(thetas):
        _X_gen[j, :] = simulator(theta, 1, random_state=rng).ravel()

    indices = rng.permutation(len(X_obs))
    _X_obs = X_obs[indices[:batch_size // 2]]
    X = np.vstack([_X_gen, _X_obs])

    y_pred = predict(X, params_critic)
    l_wgan = np.mean(-y_critic * y_pred + (1. - y_critic) * y_pred)

    # Gradient penalty
    eps = rng.rand(batch_size // 2, 1)
    _X_hat = eps * _X_obs + (1. - eps) * _X_gen
    grad_Dx = grad_predict_critic(_X_hat, params_critic)
    norms = np.sum(grad_Dx ** 2, axis=1) ** 0.5
    l_gp = np.mean((norms - 1.0) ** 2.0)

    return l_wgan + lambda_gp * l_gp

grad_loss_critic = ag.grad(loss_critic)


# grad_psi E_theta~q_psi, z~p_z(theta) [ d(g(z, theta) ]

def approx_grad_u(params_proposal, i, gamma=gamma):
    rng = check_random_state(i)
    grad_u = {k: np.zeros(len(params_proposal[k])) for k in params_proposal}
    grad_ent = {k: np.zeros(len(params_proposal[k])) for k in params_proposal}
    thetas = beta_draw(params_proposal, batch_size, random_state=rng)

    for theta in thetas:
        x = simulator(theta, 1)
        dx = predict(x, params_critic).ravel()

        grad_q = grad_beta_logpdf(params_proposal, theta)
        for k, v in grad_q.items():
            grad_u[k] += -dx * v

    grad_entropy = grad_beta_entropy(params_proposal)
    for k, v in grad_entropy.items():
        grad_ent[k] += v

    M = len(thetas)

    for k in grad_u:
        grad_u[k] = 1. / M * grad_u[k] + gamma * grad_ent[k]

    return grad_u


# Training loop

opt_critic = AdamOptimizer(grad_loss_critic, params_critic,
                           step_size=0.01, b1=0.5, b2=0.5)
opt_proposal = AdamOptimizer(approx_grad_u, params_proposal,
                             step_size=0.005, b1=0.1, b2=0.1)

opt_critic.step(100)
opt_critic.move_to(params_critic)

loss_d = []

for i in range(n_epochs):
    print(i, params_proposal)

    # fit simulator
    opt_proposal.step(1)
    opt_proposal.move_to(params_proposal)

    # fit critic
    opt_critic.reset()   # reset moments
    opt_critic.step(100)
    opt_critic.move_to(params_critic)

    loss_d.append(-loss_critic(params_critic, i, batch_size=5000))

    # plot
    if make_plots:
        fig = plt.figure(figsize=(6, 6))

        ax1 = fig.add_subplot(211)
        plt.xlim(-1.0, 1.0)
        plt.ylim(0, 1.5)
        plt.hist(X_obs, histtype="step", label=r"$x \sim p_r(x)$",
                 range=(-1.0, 1.0), bins=50, normed=1)

        thetas = beta_draw(params_proposal, 20000)
        X_gen = np.zeros((len(thetas), 1))
        for j, theta in enumerate(thetas):
            X_gen[j, :] = simulator(theta, 1).ravel()

        plt.hist(X_gen, histtype="step", label=r"$x \sim p(x|\psi)$",
                 range=(-1.0, 1.0), bins=50, normed=1)

        # thetas = np.linspace(0.0, 1.0, num=300)
        # logp = np.array([beta_logpdf(params_proposal, theta, to_scalar=False)
        #                  for theta in thetas])
        # plt.plot(thetas, np.exp([l[0] for l in logp]),
        #          label=r"$q(\log \lambda|\psi)$", linestyle="--")
        plt.legend(loc="upper right")

        ax2 = fig.add_subplot(212)
        xs = np.arange(i+1)
        plt.plot(xs, loss_d, label=r"$-U_d$")
        plt.xlim(0, n_epochs)
        plt.legend(loc="upper right")

        plt.savefig("figs/%.4d.png" % i)

        if i == n_epochs - 1:
            plt.savefig("figs/weinberg-gamma=%.2f.pdf" % gamma)

        plt.close()
