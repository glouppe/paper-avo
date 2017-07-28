# Adversarial Variational Optimization of Non-Differentiable Simulators
https://arxiv.org/abs/1707.07113

* Gilles Louppe
* Kyle Cranmer 

Complex computer simulators are increasingly used across fields of science as generative models tying parameters of an underlying theory to experimental observations. Inference in this setup is often difficult, as simulators rarely admit a tractable density or likelihood function. We introduce Adversarial Variational Optimization (AVO), a likelihood-free inference algorithm for fitting a non-differentiable generative model incorporating ideas from empirical Bayes and variational inference. We adapt the training procedure of generative adversarial networks by replacing the differentiable generative network with a domain-specific simulator. We solve the resulting non-differentiable minimax problem by minimizing variational upper bounds of the two adversarial objectives. Effectively, the procedure results in learning a proposal distribution over simulator parameters, such that the corresponding marginal distribution of the generated data matches the observations. We present results of the method with simulators producing both discrete and continuous data.

---

Please cite using the following BibTex entry:

```
@article{louppe2017avo,
    author = {{Louppe}, G. and {Cranmer}, K.},
    title = "{Adversarial Variational Optimization of Non-Differentiable Simulators}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1707.07113},
    primaryClass = "stat.ML",
    year = 2017,
    month = jul,
}

```
