"""
Analytic wake of a closed rectangular PEC cavity crossed by an
ultrarelativistic (v = c) Gaussian bunch along z.

The cavity [0, a] x [0, b] x [0, d] is the whole Wakis domain with PEC
walls; the beam enters and leaves through the end walls z = 0, d (the
closed-cavity limit of vanishing beam holes). Only the modes with E_z,

    E_z = sin(kx x) sin(ky y) cos(kz z),   kx = m pi/a, ky = n pi/b,
    kz = p pi/d (m, n >= 1, p >= 0),

couple to the beam. With k = w/c = sqrt(kx^2 + ky^2 + kz^2) and
V = int_0^d E_z(x0, y0, z) exp(i k z) dz, the modal loss factors are
k_n = |V|^2 / (4 U), U = eps0/2 int |E|^2 dV (stored energy), with

    int |E|^2 dV = abd/8 * k^2/kt^2  (p >= 1),   abd/4  (p = 0).

The wake function is w(s) = sum 2 k_n cos(k s) H(s) (k_n at s = 0, the
fundamental theorem of beam loading), so for a Gaussian bunch (rms sigma)

    W(s) = sum k_n Re[2 exp(iks - k^2 sigma^2/2) - exp(-s^2/2sigma^2) w(iz)],
    z = (s + i k sigma^2) / (sqrt(2) sigma),

with the Faddeeva function w, and the loss factor
k_loss = int W lambda ds = sum k_n exp(-k^2 sigma^2).

Units: V/C (multiply by 1e-12 for V/pC).
"""

import numpy as np
from scipy.constants import c as c_light
from scipy.constants import epsilon_0
from scipy.special import wofz


def box_modes(a, b, d, x0, y0, sigma, n_sigma=9.0):
    """
    Frequencies [rad/s] and loss factors [V/C] of the beam-coupled modes with
    k sigma <= n_sigma.

    Parameters
    ----------
    a, b, d : float
        Cavity size [m] along x, y, z.
    x0, y0 : float
        Beam position [m] in cavity coordinates (0 <= x0 <= a).
    sigma : float
        Bunch rms length [m], sets the mode cutoff k <= n_sigma / sigma.
    """
    kmax = n_sigma / sigma
    m = np.arange(1, int(kmax * a / np.pi) + 1)
    n = np.arange(1, int(kmax * b / np.pi) + 1)
    p = np.arange(0, int(kmax * d / np.pi) + 1)
    sx = np.sin(m * np.pi * x0 / a)
    sy = np.sin(n * np.pi * y0 / b)
    m, sx = m[np.abs(sx) > 1e-12], sx[np.abs(sx) > 1e-12]
    n, sy = n[np.abs(sy) > 1e-12], sy[np.abs(sy) > 1e-12]
    M, Nn, P = np.meshgrid(m, n, p, indexing="ij")
    SX, SY = np.meshgrid(sx, sy, indexing="ij")
    SX, SY = SX[..., None], SY[..., None]
    kx, ky, kz = M * np.pi / a, Nn * np.pi / b, P * np.pi / d
    kt2 = kx**2 + ky**2
    k = np.sqrt(kt2 + kz**2)
    keep = k <= kmax
    k, kz, kt2, SX, SY, P = (
        k[keep], kz[keep], kt2[keep],
        np.broadcast_to(SX, keep.shape)[keep],
        np.broadcast_to(SY, keep.shape)[keep], P[keep],
    )  # fmt: skip
    # int_0^d cos(kz z) exp(i k z) dz (k > kz: no singular denominator)
    I = 0.5 * (
        (np.exp(1j * (k + kz) * d) - 1) / (1j * (k + kz))
        + (np.exp(1j * (k - kz) * d) - 1) / (1j * (k - kz))
    )
    V2 = (SX * SY) ** 2 * np.abs(I) ** 2
    energy = np.where(P == 0, a * b * d / 4, a * b * d / 8 * k**2 / kt2)
    kn = V2 / (2 * epsilon_0 * energy)
    return k * c_light, kn


def loss_factor(omega, kn, sigma):
    """k_loss = sum k_n exp(-(w sigma/c)^2) [V/C]."""
    return float(np.sum(kn * np.exp(-((omega * sigma / c_light) ** 2))))


def wake_potential(s, omega, kn, sigma, chunk=256):
    """Gaussian-bunch wake potential W(s) [V/C] (positive = energy loss)."""
    s = np.asarray(s, float)
    W = np.zeros_like(s)
    k_all = omega / c_light
    for i in range(0, k_all.size, chunk):
        k = k_all[i : i + chunk][:, None]
        kk = kn[i : i + chunk][:, None]
        z = (s[None, :] + 1j * k * sigma**2) / (np.sqrt(2) * sigma)
        term = 2 * np.exp(1j * k * s[None, :] - 0.5 * (k * sigma) ** 2) - np.exp(
            -(s[None, :] ** 2) / (2 * sigma**2)
        ) * wofz(1j * z)
        W += np.sum(kk * term.real, axis=0)
    return W


if __name__ == "__main__":  # quick self-check against a direct convolution
    a, b, d, sig = 0.032, 0.032, 0.096, 3.2e-3
    om, kn = box_modes(a, b, d, a / 2, b / 2, sig)
    print(f"{om.size} modes, k_loss = {loss_factor(om, kn, sig) * 1e-12:.6f} V/pC")
    s = np.linspace(-6 * sig, 0.05, 2001)
    W = wake_potential(s, om, kn, sig)
    lam = np.exp(-(s**2) / (2 * sig**2)) / np.sqrt(2 * np.pi) / sig
    print(f"int W lambda ds = {np.trapezoid(W * lam, s) * 1e-12:.6f} V/pC")
