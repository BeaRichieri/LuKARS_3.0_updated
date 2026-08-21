"""
LuKARS 3.0 model functions for the Baget student example.

This module contains the numerical functions used by the accompanying
Jupyter notebook ``LuKARS3.0_Baget.ipynb``.

The purpose of this file is only to separate the model
implementation from the student-facing notebook, so that the notebook can
focus on:

- loading the input time series,
- defining the model parameters,
- running the model,
- extracting the simulated variables,
- plotting the results, and
- checking execution speed.

Notes
-----
The functions are compiled with Numba (``@njit``) for computational speed.

The Baget example
uses an hourly time step and converts simulated discharge components to
m³/s where indicated in the equations.
"""

import numpy as np
from numba import njit


@njit()
def Q_up(dt, sns, E0, Qis0, ra, kis, Qhy0, Emin, Emax, alpha, khy, lhy):
    """
    Simulate one upper LuKARS hydrotope compartment.

    The upper compartment stores water and partitions it into:
    1. infiltration from the hydrotope toward the matrix (Qis), and
    2. fast flow from the hydrotope toward the conduit (Qhy).

    Parameters
    ----------
    dt : float
        Model time-step length.
    sns : ndarray
        Input forcing time series for the hydrotope.
    E0 : float
        Initial hydrotope storage.
    Qis0 : float
        Initial infiltration flux toward the matrix.
    ra : float
        Hydrotope area.
    kis : float
        Infiltration coefficient.
    Qhy0 : float
        Initial fast-flow flux toward the conduit.
    Emin, Emax : float
        Lower and upper storage thresholds controlling fast flow.
    alpha : float
        Exponent controlling the nonlinear fast-flow response.
    khy : float
        Fast-flow coefficient.
    lhy : float
        Mean flow distance.

    Returns
    -------
    E : ndarray
        Hydrotope storage time series.
    Qis : ndarray
        Infiltration flux toward the matrix.
    Qhy : ndarray
        Fast-flow flux toward the conduit.
    """
    # Initialize variables returned by the function.
    E = np.zeros(len(sns), np.float64)
    Qis = np.zeros(len(sns), np.float64)
    Qhy = np.zeros(len(sns), np.float64)

    # Initial water levels/fluxes.
    E[0] = E0
    Qis[0] = Qis0
    Qhy[0] = Qhy0

    # Run the computation for all time steps.
    for i in range(len(sns)):  # i is the time-step index

        # Water level in the hydrotope.
        if (E[i] + (sns[i] - ((Qhy[i] + Qis[i]) / ra)) * dt) >= 0:
            E[i + 1] = E[i] + (sns[i] - ((Qhy[i] + Qis[i]) / ra)) * dt
        else:
            E[i + 1] = 0

        # Infiltration toward the matrix.
        Qis[i + 1] = ra * kis * E[i + 1]

        # Fast flow toward the conduit.
        if E[i + 1] >= Emin and Qhy[i] > 0:
            Qhy[i + 1] = (
                ((E[i + 1] - Emin) / (Emax - Emin)) ** alpha
            ) * (khy / lhy) * ra
        elif E[i + 1] >= Emax and Qhy[i] <= 0:
            Qhy[i + 1] = (
                ((E[i + 1] - Emin) / (Emax - Emin)) ** alpha
            ) * (khy / lhy) * ra
        else:
            Qhy[i + 1] = 0

    # The function output is a tuple of arrays.
    return E[:], Qis[:], Qhy[:]


@njit()
def ki_seuil(k, a, H, Hseuil):
    """
    Compute the storage-dependent recession/transfer coefficient.

    Parameters
    ----------
    k : float
        Base transfer coefficient.
    a : float
        Exponent of the nonlinear transfer law.
    H : float
        Current storage/water level.
    Hseuil : float
        Activation threshold.

    Returns
    -------
    float
        Non-negative effective transfer coefficient.
    """
    return np.maximum(k * (H - Hseuil) ** (a - 1), 0)


@njit()
def Eth(E, k, S, PAS, Emin):
    """
    Analytically update a single reservoir storage over one sub-time-step.

    Parameters
    ----------
    E : float
        Initial storage.
    k : float
        Effective recession coefficient.
    S : float
        Source term.
    PAS : float
        Length of the integration step.
    Emin : float
        Minimum allowed storage.

    Returns
    -------
    float
        Updated storage.
    """
    if k != 0:
        Eq = S / k
        return np.maximum(Eq + (E - Eq) * np.exp(-k * PAS), Emin)
    else:
        return np.maximum(E + PAS * S, Emin)


@njit()
def MCth(M, C, kMC, kM, kC, SM, SC, PAS):
    """
    Update the coupled matrix (M) and conduit (C) reservoir storages.

    This analytical solution is used by the lower-compartment calculation.
    It accounts for transfer between matrix and conduit as well as losses
    from the two reservoirs.

    Parameters
    ----------
    M, C : float
        Initial matrix and conduit storages.
    kMC : float
        Effective matrix-conduit transfer coefficient.
    kM, kC : float
        Effective matrix-spring and conduit-spring coefficients.
    SM, SC : float
        Source terms entering the matrix and conduit.
    PAS : float
        Length of the integration step.

    Returns
    -------
    Mth, Cth : float
        Updated matrix and conduit storages.
    """
    # Special case: neither lower reservoir discharges to the spring.
    if (kM == 0) & (kC == 0):
        if kMC == 0:
            Mth = M
            Cth = C
        else:
            Mth = (
                (M + C) / 2
                + (SM + SC) * PAS / 2
                + (SM - SC) / (4 * kMC)
                + (1 / 2)
                * (M - C - (SM - SC) / (2 * kMC))
                * np.exp(-2 * kMC * PAS)
            )
            Cth = (
                (M + C) / 2
                + (SM + SC) * PAS / 2
                - (SM - SC) / (4 * kMC)
                - (1 / 2)
                * (M - C - (SM - SC) / (2 * kMC))
                * np.exp(-2 * kMC * PAS)
            )

    # General coupled-reservoir case.
    else:
        kM, kC, kMC = -kM, -kC, -kMC
        f1 = np.sqrt(
            (kMC + (kC + kM) / 2) ** 2
            - (kM * kMC + kC * kMC + kC * kM)
        )
        l1 = -(kMC + (kC + kM) / 2) - f1
        l2 = -(kMC + (kC + kM) / 2) + f1

        det = kMC * kMC - (l1 + kMC + kM) * (l2 + kMC + kC)
        det_inv = 1 / det
        K100 = det_inv * kMC
        K101 = det_inv * (-l2 - kMC - kC)
        K110 = det_inv * (-l1 - kMC - kM)
        K111 = K100

        # Transformed initial conditions.
        w0 = [K100 * M + K101 * C, K110 * M + K111 * C]
        w00 = K100 * M + K101 * C
        w01 = K110 * M + K111 * C

        # Equilibrium state in transformed coordinates.
        weq0 = (K100 * SM + K101 * SC) / l1
        weq1 = (K110 * SM + K111 * SC) / l2

        # Analytical update in transformed coordinates.
        wp0 = weq0 + (w00 - weq0) * np.exp(-l1 * PAS)
        wp1 = weq1 + (w01 - weq1) * np.exp(-l2 * PAS)

        # Transform back and prevent negative storage.
        Mth = max(kMC * wp0 + (l2 + kMC + kC) * wp1, 0)
        Cth = max((l1 + kMC + kM) * wp0 + kMC * wp1, 0)

    return Mth, Cth


@njit()
def Q_bot(
    dt,
    Qis,
    Qhy,
    TotalArea,
    M0,
    C0,
    kMC,
    aMC,
    C_loss,
    M_loss,
    kMS,
    aMS,
    kCS,
    aCS,
):
    """
    Simulate the lower matrix and conduit compartments.

    Fluxes from all upper hydrotopes are aggregated and routed through the
    matrix and conduit reservoirs. The function calculates storage changes,
    matrix-conduit exchange, spring discharge, and threshold-controlled loss
    fluxes.

    Parameters
    ----------
    dt : float
        Model time-step length.
    Qis : ndarray
        Infiltration fluxes from hydrotopes toward the matrix.
    Qhy : ndarray
        Fast-flow fluxes from hydrotopes toward the conduit.
    TotalArea : float
        Total catchment area.
    M0, C0 : float
        Initial matrix and conduit storages.
    kMC, aMC : float
        Matrix-conduit transfer coefficient and exponent.
    C_loss, M_loss : float
        Storage thresholds activating conduit and matrix drainage.
    kMS, aMS : float
        Matrix-spring transfer coefficient and exponent.
    kCS, aCS : float
        Conduit-spring transfer coefficient and exponent.

    Returns
    -------
    tuple of ndarray
        QEM, QEC, Q_C_loss, Q_M_loss, Q_M_S, Q_C_S, Q_M_C,
        Q_sim, C, and M.
    """
    # Initialize variables returned by the function.
    sns_len = len(Qis)
    C = np.zeros(sns_len, np.float64)
    M = np.zeros(sns_len, np.float64)
    Q_C_loss = np.zeros(sns_len, np.float64)
    Q_M_loss = np.zeros(sns_len, np.float64)
    Q_M_S = np.zeros(sns_len, np.float64)
    Q_C_S = np.zeros(sns_len, np.float64)
    Q_M_C = np.zeros(sns_len, np.float64)
    Q_sim = np.zeros(sns_len, np.float64)

    # Initial values.
    M[0] = M0
    C[0] = C0

    # Transfer fluxes from upper to lower compartments.
    QEM = np.sum(Qis, axis=1)
    QEC = np.sum(Qhy, axis=1)

    SM = QEM / (TotalArea * 0.7)
    SC = QEC / TotalArea

    for i in range(sns_len - 1):
        if kMC == 0 or M[i] == C[i]:  # Non-coupled M-C case.
            if C[i] > C_loss:  # Drainage from C.
                Q_C_loss[i] = (C[i] - C_loss) * TotalArea / dt
                C[i] = C_loss
            if M[i] > M_loss:  # Drainage from M.
                Q_M_loss[i] = (M[i] - M_loss) * TotalArea / dt
                M[i] = M_loss

            # Non-coupled M-C: matrix compartment.
            kMSi = ki_seuil(kMS, aMS, M[i], 0)
            M12 = Eth(M[i], kMSi, SM[i], 1 / 2, 0)
            M12 = np.minimum(M12, M_loss)
            kMSi = ki_seuil(kMS, aMS, M12, 0)
            M[i + 1] = np.minimum(Eth(M[i], kMSi, SM[i], 1, 0), M_loss)
            Q_M_S[i] = np.maximum(SM[i] + (M[i] - M[i + 1]) / 1, 0)

            # Non-coupled M-C: conduit compartment.
            kCSi = ki_seuil(kCS, aCS, C[i], 0)
            C12 = Eth(C[i], kCSi, SC[i], 1 / 2, 0)
            C12 = np.minimum(C12, C_loss)  # Adjusted in original notebook.
            kCSi = ki_seuil(kCS, aCS, C12, 0)
            C[i + 1] = np.minimum(Eth(C[i], kCSi, SC[i], 1, 0), C_loss)
            Q_C_S[i] = np.maximum(SC[i] + (C[i] - C[i + 1]) / 1, 0)

        else:  # Coupled M-C case.
            if M[i] > M_loss:
                Q_M_loss[i] = (M[i] - M_loss) * TotalArea / dt
                M[i] = M_loss
            if C[i] > C_loss:
                Q_C_loss[i] = (C[i] - C_loss) * TotalArea / dt
                C[i] = C_loss

            # Effective coefficients.
            kMSi = ki_seuil(kMS, aMS, M[i], 0)
            kCSi = ki_seuil(kCS, aCS, C[i], 0)
            kMCi = ki_seuil(kMC, aMC, np.abs(M[i] - C[i]), 0)

            # Evaluate storage at t + 1/2.
            M12, C12 = MCth(
                M[i], C[i], kMCi, kMSi, kCSi, SM[i], SC[i], 1 / 2
            )

            # Update water levels.
            M12 = np.minimum(M12, M_loss)
            C12 = np.minimum(C12, C_loss)

            # Update coefficients.
            kMSi = ki_seuil(kMS, aMS, M12, 0)
            kCSi = ki_seuil(kCS, aCS, C12, 0)
            kMCi = ki_seuil(kMC, aMC, np.abs(M12 - C12), 0)

            # Evaluate storage at t + 1.
            tmpM, tmpC = MCth(
                M[i], C[i], kMCi, kMSi, kCSi, SM[i], SC[i], dt
            )

            M[i + 1] = tmpM
            C[i + 1] = tmpC

            QMSCS = (
                -(M[i + 1] - M[i])
                - (C[i + 1] - C[i])
                + SM[i]
                + SC[i]
            )

            if QMSCS == 0 or (kMSi == 0 and kCSi == 0):
                Q_M_S[i + 1] = 0
                Q_C_S[i + 1] = 0
            else:
                Q_M_S[i + 1] = QMSCS * (
                    kMSi * (M[i] + M[i + 1])
                ) / (
                    kMSi * (M[i] + M[i + 1])
                    + kCSi * (C[i] + C[i + 1])
                )
                Q_C_S[i + 1] = QMSCS - Q_M_S[i + 1]

            Q_M_C[i + 1] = (
                (M[i] - M[i + 1]) / dt + SM[i] - Q_M_S[i + 1]
            )

        Q_sim[i + 1] = np.maximum(Q_M_S[i + 1] + Q_C_S[i + 1], 0)

    # Unit conversion to m³/s.
    dt_s = 3600
    Q_C_loss = (Q_C_loss / dt_s) / 1e3
    Q_M_loss = (Q_M_loss / dt_s) / 1e3
    Q_M_S = (Q_M_S * TotalArea) / (1000 * dt_s)
    Q_C_S = (Q_C_S * TotalArea) / (1000 * dt_s)
    Q_M_C = (Q_M_C * TotalArea) / (1000 * dt_s)
    Q_sim = (Q_sim * TotalArea) / (1000 * dt_s)

    return (
        QEM,
        QEC,
        Q_C_loss,
        Q_M_loss,
        Q_M_S,
        Q_C_S,
        Q_M_C,
        Q_sim,
        C,
        M,
    )


@njit()
def run_model(
    dt,
    sns,
    E0,
    Qis0,
    Qhy0,
    ra,
    kis,
    Emin,
    Emax,
    alpha,
    khy,
    lhy,
    TotalArea,
    M0,
    C0,
    kMC,
    aMC,
    C_loss,
    M_loss,
    kMS,
    aMS,
    kCS,
    aCS,
):
    """
    Run the complete LuKARS model.

    First, ``Q_up`` is evaluated independently for every hydrotope. The
    resulting infiltration and fast-flow components are then passed to
    ``Q_bot`` to simulate the coupled lower matrix/conduit system.

    Parameters
    ----------
    dt : float
        Model time-step length.
    sns : ndarray
        Input forcing time series.
    E0, Qis0, Qhy0 : float
        Initial upper-compartment conditions.
    ra, kis, Emin, Emax, alpha, khy, lhy : ndarray
        Hydrotope-specific parameter vectors.
    TotalArea : float
        Total catchment area.
    M0, C0 : float
        Initial lower-compartment storages.
    kMC, aMC, C_loss, M_loss, kMS, aMS, kCS, aCS : float
        Lower-compartment parameters.

    Returns
    -------
    run_up : ndarray
        Upper-compartment results. Axis 0 contains E, Qis, and Qhy.
    run_bot : tuple of ndarray
        Lower-compartment results returned by ``Q_bot``.
    """
    # Time step in seconds for this hourly Baget example.
    dt_s = 3600
    nbCompartment = len(ra)

    # Run the upper model for each hydrotope.
    run_up = np.zeros(
        (3, len(sns), nbCompartment)
    )  # 3 variables returned by Q_up.

    for i in range(nbCompartment):
        E, Qis, Qhy = Q_up(
            dt,
            sns,
            E0,
            Qis0,
            ra[i],
            kis[i],
            Qhy0,
            Emin[i],
            Emax[i],
            alpha[i],
            khy[i],
            lhy[i],
        )
        run_up[0, :, i] = E
        run_up[1, :, i] = Qis
        run_up[2, :, i] = Qhy

    E = run_up[0, :, :]
    Qis = run_up[1, :, :]
    Qhy = run_up[2, :, :]

    # Run the lower model. This remains a tuple of arrays because the
    # original Numba implementation returns heterogeneous arrays as a tuple.
    run_bot = Q_bot(
        dt,
        Qis,
        Qhy,
        TotalArea,
        M0,
        C0,
        kMC,
        aMC,
        C_loss,
        M_loss,
        kMS,
        aMS,
        kCS,
        aCS,
    )

    # Unit conversion of upper-compartment fluxes to m³/s.
    Qis = (Qis / dt_s) / 1e3
    Qhy = (Qhy / dt_s) / 1e3

    run_up[1, :, :] = Qis
    run_up[2, :, :] = Qhy

    return run_up, run_bot
