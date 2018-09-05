import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt
import copy

from nn import glorot_uniform
from nn import relu
from nn import RmsPropOptimizer

from proposals import make_gaussian_proposal
from proposals import gaussian_draw
from proposals import grad_gaussian_logpdf
from proposals import grad_gaussian_entropy
from proposals import gaussian_logpdf

from sklearn.utils import check_random_state
from scipy.spatial.distance import mahalanobis

# Global params

seed = 42
rng = check_random_state(seed)

learning_rate = 10e-4
lr_schedule_rate = np.inf
batch_size = 32
n_epochs = 1000+1
lambda_reg = 20.

true_theta = np.array([(42.0-40) / (50-40),
                       (1.1-0.5) / (1.5-0.5)])

make_plots = True
plt.rcParams["figure.dpi"] = 200


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
    sqrtshalf = theta[0] * (50-40) + 40
    gf = theta[1] * (1.5 - 0.5) + 0.5

    ntrials = 0
    samples = []
    x = np.linspace(-1, 1, num=1000)
    maxval = np.max(diffxsec(x, sqrtshalf, gf))

    while len(samples) < n_samples:
        ntrials = ntrials+1
        xprop = rng.uniform(-1, 1)
        ycut = rng.rand()
        yprop = diffxsec(xprop, sqrtshalf, gf)
        if yprop/maxval < ycut:
            continue
        samples.append(xprop)

    return np.array(samples)


def simulator(theta, n_samples, random_state=None):
    rng = check_random_state(random_state)
    samples = rej_sample_costheta(n_samples, theta, rng)

    return samples.reshape(-1, 1)

X_obs = simulator(true_theta, 20000, random_state=rng)
n_params = len(true_theta)
n_features = X_obs.shape[1]


# Critic

gammas = [0.001]
colors = ["C1"]


def make_critic(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)
    params = {"W": [glorot_uniform(n_hidden, n_features, rng),
                    glorot_uniform(n_hidden, n_hidden, rng),
                    glorot_uniform(n_hidden, 0, rng, scale=0.01)],
              "b": [np.zeros(n_hidden),
                    np.zeros(n_hidden),
                    np.zeros(1)]}
    return params

params_critic = make_critic(n_features, 50, random_state=rng)


def predict(X, params):
    h = X
    h = relu(np.dot(params["W"][0], h.T).T + params["b"][0], alpha=0.2)
    h = relu(np.dot(params["W"][1], h.T).T + params["b"][1], alpha=0.2)
    h = np.dot(params["W"][2], h.T).T + params["b"][2]
    return h

grad_predict_critic = ag.elementwise_grad(predict)


history = [{"gamma": gammas[i],
            "color": colors[i],
            "loss_d": [],
            "mse": [],
            "logpdf_true": [],
            "params_proposal": make_gaussian_proposal(n_params,
                                                      mu=0.5,
                                                      log_sigma=np.log(0.1)),  # was np.log(0.1)
            "params_critic": make_critic(n_features, 50, random_state=rng)}
           for i in range(len(gammas))]


# Global loop over gammas

for state in history:
    # WGAN + GP
    def loss_critic(params_critic, i, lambda_reg=lambda_reg,
                    batch_size=batch_size):
        y_critic = np.zeros(batch_size)
        # y_critic[:batch_size // 2] = 0.0  # 0 == fake
        y_critic[batch_size // 2:] = 1.0

        rng = check_random_state(i)

        # WGAN loss
        thetas = gaussian_draw(state["params_proposal"],
                               batch_size // 2, random_state=rng)
        _X_gen = np.zeros((batch_size // 2, n_features))
        for j, theta in enumerate(thetas):
            _X_gen[j, :] = simulator(theta, 1, random_state=rng).ravel()

        indices = rng.permutation(len(X_obs))
        _X_obs = X_obs[indices[:batch_size // 2]]
        X = np.vstack([_X_gen, _X_obs])

        y_pred = predict(X, params_critic)
        l_wgan = np.mean(-y_critic * y_pred + (1. - y_critic) * y_pred)

        if lambda_reg == 0.0:
            return l_wgan

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
    def approx_grad_u(params_proposal, i):
        rng = check_random_state(i)
        grad_u = {k: np.zeros(len(params_proposal[k]))
                  for k in params_proposal}
        grad_ent = {k: np.zeros(len(params_proposal[k]))
                    for k in params_proposal}
        thetas = gaussian_draw(params_proposal, batch_size, random_state=rng)

        for theta in thetas:
            x = simulator(theta, 1, random_state=rng)
            dx = predict(x, state["params_critic"]).ravel()

            grad_q = grad_gaussian_logpdf(params_proposal, theta)
            for k, v in grad_q.items():
                grad_u[k] += -dx * v

        grad_entropy = grad_gaussian_entropy(params_proposal)
        for k, v in grad_entropy.items():
            grad_ent[k] += v

        M = len(thetas)

        for k in grad_u:
            grad_u[k] = 1. / M * grad_u[k] + state["gamma"] * grad_ent[k]

        return grad_u

    # Training loop
    init_critic = copy.copy(state["params_critic"])
    opt_critic = RmsPropOptimizer(grad_loss_critic, state["params_critic"],
                               step_size=learning_rate)
    opt_proposal = RmsPropOptimizer(approx_grad_u, state["params_proposal"],
                                 step_size=learning_rate)

    # print(predict(X_obs, state["params_critic"]).mean())
    # opt_critic.step(1000)
    # opt_critic.move_to(state["params_critic"])
    # print(predict(X_obs, state["params_critic"]).mean())

    for i in range(n_epochs):
        # fit simulator
        opt_proposal.step(1)
        opt_proposal.move_to(state["params_proposal"])

        if i > 0 and (i % lr_schedule_rate == 0):
            opt_proposal.step_size *= 0.5

        # fit critic
        opt_critic.step(1)
        opt_critic.move_to(state["params_critic"])

        # if i % 10 == 0:
        #     # # reset critic
        #     # state["params_critic"] = make_critic(n_features, 50, random_state=i)
        #     # opt_critic = AdamOptimizer(grad_loss_critic, state["params_critic"],
        #     #                            step_size=10e-4, b1=0.5, b2=0.9)
        #     # opt_critic.step(1000)
        #     # opt_critic.move_to(state["params_critic"])
        #
        #     # log
        #     state["loss_d"].append(-loss_critic(state["params_critic"], i,
        #                                         batch_size=5000))
        #
        #     print(predict(X_obs, state["params_critic"]).mean())
        #
        #     thetas = gaussian_draw(state["params_proposal"],
        #                            5000, random_state=i)
        #     _X_gen = np.zeros((5000, n_features))
        #     for j, theta in enumerate(thetas):
        #         _X_gen[j, :] = simulator(theta, 1, random_state=j).ravel()
        #     print(predict(_X_gen, state["params_critic"]).mean())
        #
        # else:
        #     state["loss_d"].append(state["loss_d"][-1])
        state["mse"].append(np.mean((true_theta - state["params_proposal"]["mu"]) ** 2))
        state["logpdf_true"].append(-gaussian_logpdf(state["params_proposal"], true_theta))

        print(i, state["gamma"], state["params_proposal"], np.mean((true_theta - state["params_proposal"]["mu"]) ** 2))

# Plot
if make_plots:
    fig = plt.figure(figsize=(6, 6))

    # proposal
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    offset = 0.15

    for state in history:
        x = np.linspace(true_theta[0]-offset, true_theta[0]+offset, num=300)
        y = np.linspace(true_theta[1]-offset, true_theta[1]+offset, num=300)
        X, Y = np.meshgrid(x, y)

        mu = state["params_proposal"]["mu"]
        sigma = np.diag(np.exp(state["params_proposal"]["log_sigma"])) ** 2.0
        sigma_inv = np.linalg.inv(sigma)

        Z = [mahalanobis(theta, mu, sigma_inv)
             for theta in np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])]
        Z = np.array(Z).reshape(X.shape)

        CS = plt.contour(X*(50-40)+40, Y*(1.5-0.5)+0.5, Z, [1.0, 2.0, 3.0], colors=state["color"])
        fmt = {l:s for l, s in zip(CS.levels, [r"$1$", r"$2$", r"$3$"])}
        plt.clabel(CS, fmt=fmt)

        plt.scatter(mu[0]*(50-40)+40, mu[1]*(1.5-0.5)+0.5, c=state["color"], marker="+")
        plt.plot([-999], [-999],
                 label=r"$q(\theta|\psi)\ \gamma=%d$" % state["gamma"],
                 color=state["color"])

    plt.scatter(true_theta[0]*(50-40)+40, true_theta[1]*(1.5-0.5)+0.5,
                c="C0",
                label=r"$\theta^* = (%d, %.1f)$" % (true_theta[0]*(50-40)+40,
                                                    true_theta[1]*(1.5-0.5)+0.5))

    plt.xlabel(r"$E_{beam}$")
    plt.ylabel(r"$G_f$")
    plt.xlim((true_theta[0]-offset)*(50-40)+40, (true_theta[0]+offset)*(50-40)+40)
    plt.ylim((true_theta[1]-offset)*(1.5-0.5)+0.5, (true_theta[1]+offset)*(1.5-0.5)+0.5)

    plt.legend(loc="lower right")

    # histograms
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    plt.xlim(-1, 1)
    Xs = [X_obs]

    for state in history:
        thetas = gaussian_draw(state["params_proposal"], 50000, random_state=rng)
        X_ = np.zeros((len(thetas), 1))
        for j, theta in enumerate(thetas):
            X_[j, :] = simulator(theta, 1).ravel()
        Xs.append(X_)

    for i in range(len(Xs)):
        Xs[i] = np.array(Xs[i]).ravel()

    plt.hist(Xs, histtype="bar",
             label=[r"$x \sim p_r(x)$"] +
                   [r"$x \sim p(x|\psi)\ \gamma=%d$" % state["gamma"] for state in history],
             color=["C0"] + [state["color"] for state in history],
             range=(-1, 1), bins=15, normed=1)
    plt.legend(loc="upper right")

    # U_g
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    xs = np.arange(n_epochs)

    for state in history:
        # plt.plot(xs,
        #          state["mse"],
        #          label=r"$-U_d\ \gamma=%d$" % state["gamma"],
        #          color=state["color"])
        plt.plot(xs,
                 state["logpdf_true"],
                 label=r"$-\log q(\theta^*|\psi)$",
                 color=state["color"])
    plt.xlim(0, n_epochs-1)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("weinberg-%d.pdf" % seed)

    plt.close()
