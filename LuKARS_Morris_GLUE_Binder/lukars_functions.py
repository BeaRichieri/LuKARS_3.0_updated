"""Utility functions for the LuKARS 3.0 hydrological-model workflow.

The functions are grouped into four sections:

1. LuKARS flow-model equations.
2. Configuration-file helpers.
3. Parameter-space transformations.
4. Output and performance-metric helpers.

"""

from __future__ import annotations

import configparser
from itertools import chain
from pathlib import Path

import numpy as np
from numba import njit


# =============================================================================
# 1. LuKARS flow-model equations
# =============================================================================

@njit()
def Q_up(
    dt,
    sns,
    E0,
    Qis0,
    ra,
    kis,
    Qhy0,
    Emin,
    Emax,
    alpha,
    khy,
    lhy,
    Qsec0,
    ksec,
    Esec,
):
    """Simulate the upper LuKARS compartment for one hydrotope.

    The function computes the hydrotope water level and the three outflow
    components:

    - ``E``: hydrotope water level/storage state.
    - ``Qis``: infiltration from the hydrotope toward the matrix.
    - ``Qhy``: fast hydrotope discharge toward the conduit.
    - ``Qsec``: secondary spring discharge.

    Returns
    -------
    tuple of numpy.ndarray
        ``(E, Qis, Qhy, Qsec)`` for the full input time series.
    """
    # Initialize variables returned by the function.
    E = np.zeros(len(sns), np.float64)
    Qis = np.zeros(len(sns), np.float64)
    Qhy = np.zeros(len(sns), np.float64)
    Qsec = np.zeros(len(sns), np.float64)

    # Initial water levels / fluxes.
    E[0] = E0
    Qis[0] = Qis0
    Qhy[0] = Qhy0
    Qsec[0] = Qsec0

    # Run the computation for every time step.
    for i in range(len(sns)):
        # Water level in the hydrotope.
        if (E[i] + (sns[i] - ((Qhy[i] + Qis[i] + Qsec[i]) / ra)) * dt) >= 0:
            E[i + 1] = E[i] + (
                sns[i] - ((Qhy[i] + Qis[i] + Qsec[i]) / ra)
            ) * dt
        else:
            E[i + 1] = 0

        # Secondary spring discharge.
        if E[i + 1] >= Esec:
            Qsec[i + 1] = ksec * ra * (E[i + 1] - Esec)
        else:
            Qsec[i + 1] = 0

        # Infiltration to the matrix.
        Qis[i + 1] = ra * kis * E[i + 1]

        # Fast flow to the conduit.
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

    return E[:], Qis[:], Qhy[:], Qsec[:]


@njit()
def ki_seuil(k, a, H, Hseuil):
    """Return the threshold-dependent recession coefficient used by LuKARS."""
    return np.maximum(k * (H - Hseuil) ** (a - 1), 0)


@njit()
def Eth(E, k, S, PAS, Emin):
    """Analytically update a storage state over one integration sub-step."""
    if k != 0:
        Eq = S / k
        return np.maximum(Eq + (E - Eq) * np.exp(-k * PAS), Emin)
    return np.maximum(E + PAS * S, Emin)


@njit()
def MCth(M, C, kMC, kM, kC, SM, SC, PAS):
    """Update coupled matrix and conduit storage states analytically.

    This is the matrix-conduit integration routine used by ``Q_bot``. The
    equations and branch structure are unchanged from the original notebook.
    """
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

        w00 = K100 * M + K101 * C
        w01 = K110 * M + K111 * C

        weq0 = (K100 * SM + K101 * SC) / l1
        weq1 = (K110 * SM + K111 * SC) / l2

        wp0 = weq0 + (w00 - weq0) * np.exp(-l1 * PAS)
        wp1 = weq1 + (w01 - weq1) * np.exp(-l2 * PAS)

        # Preserve the original non-negative storage condition.
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
    """Simulate the lower matrix and conduit compartments.

    The function receives the aggregated upper-compartment fluxes and
    computes matrix/conduit states, losses, exchange fluxes and spring
    discharge. Unit conversions to m3/s.

    Returns
    -------
    tuple of numpy.ndarray
        ``(QEM, QEC, Q_C_loss, Q_M_loss, Q_M_S, Q_C_S, Q_M_C,
        Q_sim, C, M)``.
    """
    sns_len = len(Qis)

    # State and flux arrays.
    C = np.zeros(sns_len, np.float64)
    M = np.zeros(sns_len, np.float64)
    Q_C_loss = np.zeros(sns_len, np.float64)
    Q_M_loss = np.zeros(sns_len, np.float64)
    Q_M_S = np.zeros(sns_len, np.float64)
    Q_C_S = np.zeros(sns_len, np.float64)
    Q_M_C = np.zeros(sns_len, np.float64)
    Q_sim = np.zeros(sns_len, np.float64)

    # Initial storage states.
    M[0] = M0
    C[0] = C0

    # Transfer of fluxes from upper to lower compartments.
    QEM = np.sum(Qis, axis=1)
    QEC = np.sum(Qhy, axis=1)

    SM = QEM / TotalArea
    SC = QEC / TotalArea

    for i in range(sns_len - 1):
        if kMC == 0 or M[i] == C[i]:
            # -------------------------------------------------------------
            # Non-coupled matrix and conduit compartments.
            # -------------------------------------------------------------
            if C[i] > C_loss:
                Q_C_loss[i] = (C[i] - C_loss) * TotalArea / dt
                C[i] = C_loss

            if M[i] > M_loss:
                Q_M_loss[i] = (M[i] - M_loss) * TotalArea / dt
                M[i] = M_loss

            # Matrix compartment.
            kMSi = ki_seuil(kMS, aMS, M[i], 0)
            M12 = Eth(M[i], kMSi, SM[i], 1 / 2, 0)
            M12 = np.minimum(M12, M_loss)
            kMSi = ki_seuil(kMS, aMS, M12, 0)
            M[i + 1] = np.minimum(Eth(M[i], kMSi, SM[i], 1, 0), M_loss)
            Q_M_S[i] = np.maximum(SM[i] + (M[i] - M[i + 1]) / 1, 0)

            # Conduit compartment.
            kCSi = ki_seuil(kCS, aCS, C[i], 0)
            C12 = Eth(C[i], kCSi, SC[i], 1 / 2, 0)
            C12 = np.minimum(C12, C_loss)
            kCSi = ki_seuil(kCS, aCS, C12, 0)
            C[i + 1] = np.minimum(Eth(C[i], kCSi, SC[i], 1, 0), C_loss)
            Q_C_S[i] = np.maximum(SC[i] + (C[i] - C[i + 1]) / 1, 0)

        else:
            # -------------------------------------------------------------
            # Coupled matrix-conduit compartments.
            # -------------------------------------------------------------
            if M[i] > M_loss:
                Q_M_loss[i] = (M[i] - M_loss) * TotalArea / dt
                M[i] = M_loss

            if C[i] > C_loss:
                Q_C_loss[i] = (C[i] - C_loss) * TotalArea / dt
                C[i] = C_loss

            # Coefficients at t.
            kMSi = ki_seuil(kMS, aMS, M[i], 0)
            kCSi = ki_seuil(kCS, aCS, C[i], 0)
            kMCi = ki_seuil(kMC, aMC, np.abs(M[i] - C[i]), 0)

            # Evaluate storage at t + 1/2.
            M12, C12 = MCth(
                M[i], C[i], kMCi, kMSi, kCSi, SM[i], SC[i], 1 / 2
            )

            M12 = np.minimum(M12, M_loss)
            C12 = np.minimum(C12, C_loss)

            # Update coefficients at t + 1/2.
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

    # Unit conversion to m3/s.
    dt_s = 86400
    QEM = (QEM / dt_s) / 1e3
    QEC = (QEC / dt_s) / 1e3
    Q_C_loss = (Q_C_loss / dt_s) / 1e3
    Q_M_loss = (Q_M_loss / dt_s) / 1e3
    Q_M_S = (Q_M_S * TotalArea) / (1000 * dt_s)
    Q_C_S = (Q_C_S * TotalArea) / (1000 * dt_s)
    Q_M_C = (Q_M_C * TotalArea) / (1000 * dt_s)
    Q_sim = np.maximum(Q_C_S + Q_M_S, 0)

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
    sns_0,
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
    Qsec0,
    ksec,
    Esec,
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
    """Run the complete LuKARS model for one parameter realization.

    Each hydrotope is simulated with ``Q_up``. The resulting infiltration and
    fast-flow components are then passed to ``Q_bot``. Upper-compartment
    discharge components are converted to m3/s before being returned.
    """
    dt_s = 86400
    nbCompartment = len(ra)

    # Four variables are returned by Q_up for each hydrotope.
    run_up = np.zeros((4, len(sns), nbCompartment))

    for i in range(nbCompartment):
        # Hydrotope 1 uses sns_0; all remaining hydrotopes use sns.
        if i == 0:
            current_sns = sns_0
        else:
            current_sns = sns

        E, Qis, Qhy, Qsec = Q_up(
            dt,
            current_sns,
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
            Qsec0,
            ksec[i],
            Esec[i],
        )
        run_up[0, :, i] = E
        run_up[1, :, i] = Qis
        run_up[2, :, i] = Qhy
        run_up[3, :, i] = Qsec

    Qis = run_up[1, :, :]
    Qhy = run_up[2, :, :]

    # Run lower compartments. A tuple is retained for Numba compatibility.
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

    # Convert upper-compartment discharge components to m3/s.
    run_up[1, :, :] = (run_up[1, :, :] / dt_s) / 1e3
    run_up[2, :, :] = (run_up[2, :, :] / dt_s) / 1e3
    run_up[3, :, :] = (run_up[3, :, :] / dt_s) / 1e3

    return run_up, run_bot


# =============================================================================
# 2. Configuration-file helpers
# =============================================================================

def parse_value(val):
    """Parse one value read from ``config.ini``.

    The configuration allows arithmetic expressions such as
    ``2.5 * 1e6`` and NumPy expressions. 
    """
    try:
        return eval(val, {"np": np})
    except Exception:
        try:
            return float(val)
        except ValueError:
            return val


def read_config(config_path):
    """Read ``config.ini`` while preserving the case of parameter names.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the configuration file.

    Returns
    -------
    settings : dict
        Flat dictionary whose keys use ``Section.parameter`` notation.
    config : configparser.ConfigParser
        Parsed configuration object, retained so hydrotope sections can be
        detected dynamically in the notebook.
    """
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case of keys such as kMC and C_loss.
    config.read(config_path)

    settings = {}
    for section in config.sections():
        for key, val in config.items(section):
            compound_key = f"{section}.{key}"
            settings[compound_key] = parse_value(val)

    return settings, config


# =============================================================================
# 3. Parameter-space transformations
# =============================================================================

def fromCalibrationToInterimParameters(samples, lower_bounds, upper_bounds):
    """Map normalized Morris samples from [-1, 1] to interim parameter space.

    ``lower_bounds`` and ``upper_bounds`` are the transformed parameter bounds
    assembled from ``config.ini`` by the notebook. Passing them explicitly
    avoids hidden module-level state while preserving the original linear
    transformation.
    """
    return (
        0.5 * (upper_bounds - lower_bounds) * samples
        + 0.5 * (upper_bounds + lower_bounds)
    )


def fromInterimToCalibrationParameters(samples, lower_bounds, upper_bounds):
    """Map interim parameter values back to normalized calibration space."""
    return (
        2.0 / (upper_bounds - lower_bounds) * samples
        - (upper_bounds + lower_bounds) / (upper_bounds - lower_bounds)
    )


def fromCalibrationToPhysicalParameters(
    samples, lower_bounds, upper_bounds, hydrotope_ids
):
    """Convert normalized samples to physically meaningful LuKARS parameters.

    The transformation consists of:

    - log-scaled parameters are exponentiated;
    - ``e_max`` is reconstructed as ``e_min + diff_e``;
    - seven parameters are reconstructed for each hydrotope; and
    - ``kMS`` is reconstructed from the final sample column.

    Parameters
    ----------
    samples : numpy.ndarray
        Morris samples in normalized calibration space.
    lower_bounds, upper_bounds : numpy.ndarray
        Transformed lower and upper bounds read from ``config.ini``.
    hydrotope_ids : sequence of int
        Hydrotope identifiers detected from the config sections.

    Returns
    -------
    dict
        Physical parameter arrays keyed by parameter name.
    """
    samples = np.array(samples)
    if len(samples.shape) <= 1:
        samples = samples[:, np.newaxis]

    samples = fromCalibrationToInterimParameters(
        samples, lower_bounds, upper_bounds
    )
    physical_params = {}

    for i, h in enumerate(hydrotope_ids):
        base_idx = i * 7
        physical_params[f"k_e_{h}_num"] = np.exp(samples[:, base_idx])
        physical_params[f"e_min_{h}"] = samples[:, base_idx + 1]
        physical_params[f"e_max_{h}"] = (
            physical_params[f"e_min_{h}"] + samples[:, base_idx + 2]
        )
        physical_params[f"alpha_{h}"] = samples[:, base_idx + 3]
        physical_params[f"k_is_{h}"] = np.exp(samples[:, base_idx + 4])
        physical_params[f"k_sec_{h}"] = np.exp(samples[:, base_idx + 5])
        physical_params[f"e_sec_{h}"] = samples[:, base_idx + 6]

    physical_params["kMS"] = np.exp(samples[:, -1])
    return physical_params


def fromPhysicalToCalibrationParameters(
    physical_samples, lower_bounds, upper_bounds, hydrotope_ids
):
    """Convert physical LuKARS parameter arrays back to calibration space.

    This inverse transformation is required by SALib's Morris analysis because
    the analysis must receive the same normalized sample design used to create
    the physical model realizations.
    """
    samples = np.hstack(
        list(
            chain.from_iterable(
                [
                    [
                        np.log(physical_samples[f"k_e_{h}_num"])[
                            :, np.newaxis
                        ],
                        physical_samples[f"e_min_{h}"][:, np.newaxis],
                        (
                            physical_samples[f"e_max_{h}"]
                            - physical_samples[f"e_min_{h}"]
                        )[:, np.newaxis],
                        physical_samples[f"alpha_{h}"][:, np.newaxis],
                        np.log(physical_samples[f"k_is_{h}"])[
                            :, np.newaxis
                        ],
                        np.log(physical_samples[f"k_sec_{h}"])[
                            :, np.newaxis
                        ],
                        physical_samples[f"e_sec_{h}"][:, np.newaxis],
                    ]
                    for h in hydrotope_ids
                ]
            )
        )
        + [np.log(physical_samples["kMS"])[:, np.newaxis]]
    )

    return fromInterimToCalibrationParameters(
        samples, lower_bounds, upper_bounds
    )


# =============================================================================
# 4. Output and performance-metric helpers
# =============================================================================

def save_output(filename, data, output_dir="Output"):
    """Save a one-dimensional parameter/output array as a tab-delimited file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path / f"{filename}.txt",
        np.array(data).reshape(-1, 1),
        delimiter="\t",
    )


def nash_sutcliffe_efficiency(observed, simulated):
    """Calculate the Nash-Sutcliffe efficiency (NSE).

    The implementation is identical to the original notebook and returns
    ``1 - sum((obs-sim)^2) / sum((obs-mean(obs))^2)``.
    """
    observed = np.array(observed)
    simulated = np.array(simulated)

    if len(observed) != len(simulated):
        raise ValueError(
            "The observed and simulated arrays must have the same length."
        )

    mean_observed = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    nse = numerator / denominator

    return 1 - nse
