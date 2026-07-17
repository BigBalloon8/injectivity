from __future__ import annotations

import numpy as np
from scipy.linalg import null_space


def project_kernel_into_safe_region(
    M: np.ndarray,
    A: np.ndarray,
    gamma_lower: float,
    safety_factor: float = 0.99,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Project ker(M) into a certified safe gap ball around col(A)^perp.

    Parameters
    ----------
    M:
        Matrix of shape (n, m). It should have full row rank n.

    A:
        Matrix of shape (m, n). It should have full column rank n.

    gamma_lower:
        A certified lower bound on gamma_0(A, c).
        It must satisfy 0 < gamma_lower <= gamma_0.

    safety_factor:
        Number strictly between 0 and 1. The target gap is
        safety_factor * gamma_lower.

    tolerance:
        Numerical tolerance.

    Returns
    -------
    M_safe:
        A matrix whose kernel lies strictly inside the certified safe region.

    U_safe:
        An orthonormal basis for ker(M_safe), with shape (m, m-n).

    diagnostics:
        Original and final gap information.
    """
    M = np.asarray(M, dtype=float)
    A = np.asarray(A, dtype=float)

    if M.ndim != 2 or A.ndim != 2:
        raise ValueError("M and A must both be matrices.")

    n, m = M.shape

    if A.shape != (m, n):
        raise ValueError(
            f"Expected A to have shape {(m, n)}, but got {A.shape}."
        )

    if np.linalg.matrix_rank(A, tol=tolerance) != n:
        raise ValueError("A must have full column rank.")

    if np.linalg.matrix_rank(M, tol=tolerance) != n:
        raise ValueError(
            "M must have full row rank. Otherwise dim ker(M) is larger "
            "than dim col(A)^perp, and their projector gap is 1."
        )

    if not 0.0 < gamma_lower <= 1.0:
        raise ValueError("gamma_lower must lie in (0, 1].")

    if not 0.0 < safety_factor < 1.0:
        raise ValueError("safety_factor must lie strictly between 0 and 1.")

    # Orthonormal bases for K = ker(M) and K0 = ker(A^T).
    U = null_space(M, rcond=tolerance)
    U0 = null_space(A.T, rcond=tolerance)

    if U.shape != U0.shape:
        raise RuntimeError(
            f"Kernel dimensions disagree: {U.shape[1]} versus {U0.shape[1]}."
        )

    # Principal-angle decomposition:
    #
    # U0.T @ U = L diag(cos(theta_i)) R.T
    L, cos_theta, Rt = np.linalg.svd(
        U0.T @ U,
        full_matrices=False,
    )

    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    theta = np.arccos(cos_theta)

    R = Rt.T

    # Corresponding principal vectors.
    P = U0 @ L       # Principal vectors in K0
    Q = U @ R        # Principal vectors in K

    target_gap = safety_factor * gamma_lower
    theta_cap = np.arcsin(target_gap)

    Q_safe = np.empty_like(Q)

    for i, angle in enumerate(theta):
        # This direction is already safe.
        if angle <= theta_cap:
            Q_safe[:, i] = Q[:, i]
            continue

        sin_angle = np.sin(angle)

        if sin_angle <= tolerance:
            # This should not occur when angle > theta_cap >= 0,
            # but it protects against numerical problems.
            Q_safe[:, i] = P[:, i]
            continue

        # Unit direction orthogonal to K0 in the principal two-plane.
        residual = (
            Q[:, i] - np.cos(angle) * P[:, i]
        ) / sin_angle

        Q_safe[:, i] = (
            np.cos(theta_cap) * P[:, i]
            + np.sin(theta_cap) * residual
        )

    # Symmetric orthonormalisation to correct floating-point drift.
    gram = Q_safe.T @ Q_safe
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    if np.min(eigenvalues) <= tolerance:
        raise RuntimeError("Projected kernel basis became numerically singular.")

    inverse_sqrt_gram = (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    U_safe = Q_safe @ inverse_sqrt_gram

    P0 = U0 @ U0.T
    P_original = U @ U.T
    P_safe = U_safe @ U_safe.T

    original_gap = np.linalg.norm(P_original - P0, ord=2)
    final_gap = np.linalg.norm(P_safe - P0, ord=2)

    # Orthogonal projection of each row of M onto ker(U_safe.T).
    M_safe = M @ (np.eye(m) - P_safe)

    final_rank = np.linalg.matrix_rank(M_safe, tol=tolerance)

    if final_rank != n:
        raise RuntimeError(
            "The projected matrix lost row rank numerically. "
            "Use the orthonormal-row construction described below."
        )

    annihilation_error = np.linalg.norm(M_safe @ U_safe, ord=2)

    diagnostics = {
        "principal_angles_radians": theta,
        "principal_angles_degrees": np.degrees(theta),
        "angle_cap_radians": theta_cap,
        "angle_cap_degrees": np.degrees(theta_cap),
        "original_gap": original_gap,
        "target_gap": target_gap,
        "final_gap": final_gap,
        "gamma_lower": gamma_lower,
        "matrix_change_frobenius": np.linalg.norm(M_safe - M, ord="fro"),
        "annihilation_error": annihilation_error,
        "final_rank": final_rank,
    }

    return M_safe, U_safe, diagnostics

if __name__ == "__main__":
    import torch
    A = torch.randn(16, 4)
    b = torch.randn(16)
    B = torch.randn(4, 16)
    
    M_safe, K_safe_basis, info = project_kernel_into_safe_region(
        M=B,
        A=A,
        gamma_lower=0.1,
        safety_factor=0.95,
    )

    print("Original gap:", info["original_gap"])
    print("Final gap:", info["final_gap"])
    print("Certified threshold:", info["gamma_lower"])