import numpy as np
from scipy.spatial import distance
from scipy import stats 
from scipy.stats import wasserstein_distance

def Euclidean_Distances(A, B):
    return distance.euclidean(A, B)


def Cosine_Distances(A, B):
    return distance.cosine(A, B)


def Pearson_Distances(A, B):
    A = np.squeeze(A)
    B = np.squeeze(B)
    r, p = stats.pearsonr(A, B)
    return 1-r, p


def KL_Divergence(A_mean, A_logvar, B_mean, B_logvar):
    kl_AB = np.sum(B_logvar) - np.sum(A_logvar) + np.sum((1/np.exp(B_logvar) * np.exp(A_logvar))) \
            + np.sum((B_mean - A_mean) * B_logvar * (B_mean - A_mean)) - A_mean.shape[0]
    kl_BA = np.sum(A_logvar) - np.sum(B_logvar) + np.sum((1/np.exp(A_logvar) * np.exp(B_logvar))) \
            + np.sum((A_mean - B_mean) * A_logvar * (A_mean - B_mean)) - A_mean.shape[0]
    kl_divergence = 0.25 * (kl_AB + kl_BA)
    return kl_divergence



def Wasserstein_Distance(A_mean, A_logvar, B_mean, B_logvar):
    p1 = (np.sum(np.subtract(A_mean, B_mean) ** 2)) ** 0.5
    p2 = np.sum(np.subtract(np.exp(A_logvar) ** 0.5, np.exp(B_logvar) ** 0.5) ** 2)
    return p1+p2



def npd(
        x: np.ndarray, y: np.ndarray,
        x_posterior: np.ndarray, y_posterior: np.ndarray, eps: float = 0.0
) -> np.ndarray:  # pragma: no cover
    r"""
    x : latent_dim
    y : latent_dim
    x_posterior : n_posterior * latent_dim
    y_posterior : n_posterior * latent_dim
    """
    projection = x - y  # latent_dim
    if np.all(projection == 0.0):
        projection[...] = 1.0  # any projection is equivalent
    projection /= np.linalg.norm(projection)
    x_posterior = np.sum(x_posterior * projection, axis=1)  # n_posterior_samples
    y_posterior = np.sum(y_posterior * projection, axis=1)  # n_posterior_samples
    xy_posterior = np.concatenate((x_posterior, y_posterior))
    xy_posterior1 = (xy_posterior - np.mean(x_posterior)) / (np.std(x_posterior) + np.float32(eps))
    xy_posterior2 = (xy_posterior - np.mean(y_posterior)) / (np.std(y_posterior) + np.float32(eps))
    return 0.5 * (wasserstein_distance(xy_posterior1[:len(x_posterior)],
    xy_posterior1[-len(y_posterior):])
     + wasserstein_distance(
        xy_posterior2[:len(x_posterior)],
        xy_posterior2[-len(y_posterior):]
    ))