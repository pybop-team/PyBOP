import matplotlib.pyplot as plt
from numpy import array, exp, log, logspace
from scipy.stats import lognorm

"""
Suppose a normally distributed random variable ``X`` has  mean ``mu``
and standard deviation ``sigma``. Then ``Y = exp(X)`` is lognormally
distributed with ``s = sigma`` and ``scale = exp(mu)``.
"""


def calculate_distribution_parameters(log_evidence, log_variance):
    sigma = log(1 + exp(log_variance) / exp(log_evidence) ** 2) ** 0.5
    mu = log_evidence - sigma**2 / 2
    mean = exp(log_evidence)
    lower = mean / exp(sigma)
    upper = mean * exp(sigma)
    print(mean, "within one confidence interval: [", lower, ",", upper, "]")
    return mu, sigma


def plot_evidence_distribution(log_evidences, log_variances, labels, title):
    fig, ax = plt.subplots(figsize=(3**0.5, 3))
    log_evidences = array(log_evidences)
    log_variances = array(log_variances)
    min_extent = float("inf")
    max_extent = -float("inf")
    for le, lv, label in zip(log_evidences, log_variances, labels):
        mu, sigma = calculate_distribution_parameters(le, lv)
        lower = mu - 2 * sigma
        upper = mu + 2 * sigma
        evidence_eval = logspace(lower, upper, 200, base=exp(1))
        min_extent = min(lower, min_extent)
        max_extent = max(upper, max_extent)
        distribution = lognorm(s=sigma, scale=exp(mu))
        distribution_eval = distribution.pdf(evidence_eval)
        ax.plot(evidence_eval, distribution_eval, label=label)
    ax.set_xlim(0.9 * exp(min_extent), 1.1 * exp(max_extent))
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# SPM impedance results.
"""
plot_evidence_distribution(
    [-4.86830e+00, -5.14230e+00, -5.45996e+00],
    [-3.39258e+00, -9.72431e+00, -6.52256e+00],
    ["17 μm", "42 μm", "80 μm"],
    "SPM impedance\n89 % delithiation"
)
"""
plot_evidence_distribution(
    [-7.06211e-01, -4.96983e00, -3.85247e00],
    [-1.56320e00, -5.19572e00, -4.24408e00],
    ["17 μm", "42 μm", "80 μm"],
    "SPM impedance\n45 % lithiation",
)

# DFN impedance results.
"""
plot_evidence_distribution(
    [-8.30258e+00, -7.74164e+00, -6.14714e+00],
    [-1.06599e+01, -1.76625e+01, -8.05868e+00],
    ["17 μm", "42 μm", "80 μm"],
    "DFN impedance\n89 % delithiation"
)
"""

plot_evidence_distribution(
    [-1.55120e00, -8.44433e00, -6.67521e00],
    [-1.04638e01, -8.70146e00, -8.58370e00],
    ["17 μm", "42 μm", "80 μm"],
    "DFN impedance\n45 % lithiation",
)

# SEI impedance results.
plot_evidence_distribution(
    [-2.51683e00, -9.07830e00],
    [-5.38383e00, -9.82745e00],
    ["τ_SEI > τ_DL", "τ_SEI < τ_DL"],
    "SEI model impedance",
)

# Knee point model selection.
calculate_distribution_parameters(-3.00752, -15.8633)
calculate_distribution_parameters(-1.66418, -19.5028)
