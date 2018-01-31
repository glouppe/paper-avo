import autograd as ag
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from nn import glorot_uniform
from nn import relu
from nn import AdamOptimizer

from proposals import make_gaussian_proposal
from proposals import gaussian_draw
from proposals import gaussian_logpdf
from proposals import grad_gaussian_logpdf
from proposals import grad_gaussian_entropy

from sklearn.utils import check_random_state

# Global params

seed = 777
rng = check_random_state(seed)

batch_size = 64   # try increasing, default=64
n_epochs = 500+1
lambda_gp = 0.1 # ok~0.1; try reducing (=> worse), try increasing (0.4, not sure what to conclude)

# see zigzag near the end... hence the reset? but works better without in multi.py

true_theta = np.array([np.log(7)])
make_plots = True


# Simulator

def simulator(theta, n_samples, random_state=None):
    return poisson.rvs(np.exp(theta[0]),
                       size=n_samples,
                       random_state=random_state).reshape(-1, 1)

X_obs = simulator(true_theta, 20000, random_state=rng)
n_params = len(true_theta)
n_features = X_obs.shape[1]


# Critic

gammas = [0.0, 5.0]
colors = ["C1", "C2"]

# gammas = [5.0]
# colors = ["C2"]

def make_critic(n_features, n_hidden, random_state=None):
    rng = check_random_state(random_state)
    params = {"W": [glorot_uniform(n_hidden, n_features, rng),
                    glorot_uniform(n_hidden, n_hidden, rng),
                    glorot_uniform(n_hidden, 0, rng)],
              "b": [np.zeros(n_hidden),
                    np.zeros(n_hidden),
                    np.zeros(1)]}
    return params

def predict(X, params):
    h = X
    h = relu(np.dot(params["W"][0], h.T).T + params["b"][0], alpha=0.1)
    h = relu(np.dot(params["W"][1], h.T).T + params["b"][1], alpha=0.1)
    h = np.dot(params["W"][2], h.T).T + params["b"][2]
    return h

grad_predict_critic = ag.elementwise_grad(predict)


history = [{"gamma": gammas[i],
            "color": colors[i],
            "loss_d": [],
            "params_proposal": make_gaussian_proposal(n_params,
                                                      mu=np.log(10.0)),
            "params_critic": make_critic(n_features, 10, random_state=rng)}
           for i in range(len(gammas))]


# Global loop over gammas

for state in history:
    # WGAN + GP
    def loss_critic(params_critic, i, lambda_gp=lambda_gp, batch_size=batch_size):
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

        # Gradient penalty
        eps = rng.rand(batch_size // 2, 1)
        _X_hat = eps * _X_obs + (1. - eps) * _X_gen
        grad_Dx = grad_predict_critic(_X_hat, params_critic)
        norms = np.sum(grad_Dx ** 2, axis=1) ** 0.5
        l_gp = np.mean((norms - 1.0) ** 2.0)

        return l_wgan + lambda_gp * l_gp

    grad_loss_critic = ag.grad(loss_critic)

    # grad_psi E_theta~q_psi, z~p_z(theta) [ d(g(z, theta)
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
    opt_critic = AdamOptimizer(grad_loss_critic, state["params_critic"], step_size=0.005, b1=0.05, b2=0.05) #, step_size=0.01, b1=0.5, b2=0.5)
    opt_proposal = AdamOptimizer(approx_grad_u, state["params_proposal"], step_size=0.005, b1=0.05, b2=0.05) #, step_size=0.01, b1=0.5, b2=0.5)

    opt_critic.step(100)
    opt_critic.move_to(state["params_critic"])
    opt_critic.reset()

    for i in range(n_epochs):
        print(i, state["gamma"], state["params_proposal"], np.mean((true_theta - state["params_proposal"]["mu"]) ** 2))

        # fit simulator
        #opt_proposal.reset()
        opt_proposal.step(1)
        opt_proposal.move_to(state["params_proposal"])

        # fit critic
        #opt_critic.reset()   # reset moments
        opt_critic.step(10)
        opt_critic.move_to(state["params_critic"])

        state["loss_d"].append(-loss_critic(state["params_critic"], i,
                                            batch_size=5000))


# Plot

if make_plots:
    fig = plt.figure(figsize=(6, 6))

    # proposal
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    plt.xlim(true_theta[0]-1, true_theta[0]+2)

    plt.axvline(x=true_theta[0], linestyle="--", label=r"$\theta^*$")
    thetas = np.linspace(true_theta[0]-1, true_theta[0]+2, num=300)

    for state in history:
        logp = np.array([gaussian_logpdf(state["params_proposal"],
                                         theta, to_scalar=False)
                         for theta in thetas])
        plt.plot(thetas,
                 np.exp([l[0] for l in logp]),
                 label=r"$q(\theta|\psi)\ \gamma=%d$" % state["gamma"],
                 color=state["color"])

    plt.legend(loc="upper right")
    plt.ylim(0, 6)

    # histograms
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    plt.xlim(0, 15)
    plt.ylim(0, 0.25)

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
             range=(0, 15), bins=16, normed=1)
    plt.legend(loc="upper right")

    # U_g
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    xs = np.arange(n_epochs)

    for state in history:
        plt.plot(xs,
                 state["loss_d"],
                 label=r"$-U_d\ \gamma=%d$" % state["gamma"],
                 color=state["color"])
    plt.xlim(0, n_epochs)
    plt.ylim(0, 10)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("poisson-%d.pdf" % seed)

    plt.close()
