"""Force estimation port of external_force.mlx with empirical confidence bands.

The original MATLAB live script computes external wrench from

    force = inv(J.T) @ (measured_joint_torque - predicted_free_space_torque)

then rotates the first three force axes and applies a moving average.

This Python version keeps that pipeline and adds a leak-free 95% interval. The
force sensor is used only for validation/RMSE, never to set the interval. The
interval propagates torque-side uncertainty through J^-T and inflates that
uncertainty when estimated contact force is small or when the real-time
observer disagrees with the immediate torque-to-force projection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter


CHANNELS = ("Fx", "Fy", "Fz", "Taux", "Tauy", "Tauz")
JOINT_VELOCITY_CHANNELS = ("q1_dot", "q2_dot", "q3_dot", "q4_dot", "q5_dot", "q6_dot")
CARTESIAN_VELOCITY_CHANNELS = ("vx", "vy", "vz", "wx", "wy", "wz")


@dataclass
class ForceResults:
    time: np.ndarray
    external_torque: np.ndarray
    raw_force: np.ndarray
    force: np.ndarray
    real_force: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_half_width: np.ndarray
    rmse: np.ndarray
    sigma_bin_centers: np.ndarray
    sigma_by_channel: np.ndarray
    joint_velocity: np.ndarray
    cartesian_velocity: np.ndarray
    cartesian_speed: np.ndarray
    estimator_mode: str
    observer_gain: float
    bias_gain: float
    deadband_multiplier: float


def read_matrix(path: Path) -> np.ndarray:
    """Read MATLAB readmatrix-style numeric CSV or whitespace tables."""
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    delimiter = "," if "," in first_line else None
    return np.loadtxt(path, delimiter=delimiter)


def default_data_root() -> Path:
    script_dir = Path(r"D:\_RESEARCH\paper_plot_scripts\HSMR_2024_Effectiveness")
    candidates = [
        script_dir.parent / "dvrk_si_col_9_1",
        Path.cwd().parent / "dvrk_si_col_9_1",
        Path.cwd() / "dvrk_si_col_9_1",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return lfilter(kernel, [1.0], data, axis=0)


def centered_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Centered moving average for presentation-only diagnostic plots."""
    if window_size <= 1:
        return data
    pad_before = window_size // 2
    pad_after = window_size - 1 - pad_before
    padded = np.pad(data, [(pad_before, pad_after)] + [(0, 0)] * (data.ndim - 1), mode="edge")
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)


def build_jacobian(jacobian_data: np.ndarray, n_samples: int) -> np.ndarray:
    """Rebuild the MATLAB 6x6xN Jacobian tensor."""
    # MATLAB: jacobian_data(:, 2:37).' -> reshape(...,[6,6,length]) -> permute([2 1 3])
    return (
        jacobian_data[:, 1:37]
        .T.reshape(6, 6, -1, order="F")
        .transpose(1, 0, 2)[:, :, :n_samples]
    )


def force_rotation_matrix() -> np.ndarray:
    """Rotation applied to the first three force channels in the MATLAB script."""
    angle = np.pi / 2.0
    ra = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rz = np.array(
        [
            [-1.0000, 0.0, 0.0],
            [0.0, -0.7071, -0.7071],
            [0.0, -0.7071, 0.7071],
        ]
    )
    return ra @ rz


def torque_error_from_data(joint_data: np.ndarray, torque_data: np.ndarray) -> np.ndarray:
    n_samples = torque_data.shape[0]
    predicted_torque = torque_data[:, 1:7].T
    measured_torque = joint_data[:n_samples, 13:19].T
    return measured_torque - predicted_torque


def residual_observer(
    torque_error: np.ndarray,
    time: np.ndarray,
    observer_gain: float | np.ndarray,
) -> np.ndarray:
    """Apply r_dot = K_i (tau_ext_hat - r) to the LSTM torque residual.

    torque_error is shaped 6 x N and acts as tau_ext_hat. The exact discrete
    first-order update avoids gain-dependent Euler stability issues:
    r_k = exp(-K_i dt) r_{k-1} + (1 - exp(-K_i dt)) tau_ext_hat_k.
    """
    gain = np.asarray(observer_gain, dtype=float)
    if gain.ndim == 0:
        gain = np.full(torque_error.shape[0], float(gain))
    if gain.shape[0] != torque_error.shape[0]:
        raise ValueError("observer_gain must be scalar or length 6")

    observed = np.empty_like(torque_error)
    observed[:, 0] = torque_error[:, 0]
    dt = np.diff(time[: torque_error.shape[1]])
    dt = np.maximum(dt, 0.0)
    for idx, sample_dt in enumerate(dt, start=1):
        alpha = np.exp(-gain * sample_dt)
        observed[:, idx] = alpha * observed[:, idx - 1] + (1.0 - alpha) * torque_error[:, idx]
    return observed


def soft_deadband(values: np.ndarray, deadband: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - deadband[:, None], 0.0)


def smooth_deadband(values: np.ndarray, deadband: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Smoothly suppress small torque residuals without subtracting a hard offset."""
    width = np.maximum(width, 1e-12)
    gate = 1.0 / (1.0 + np.exp(-(np.abs(values) - deadband[:, None]) / width[:, None]))
    return values * gate


def parse_six_vector(value: str | Iterable[float] | None, default: Iterable[float]) -> np.ndarray:
    if value is None:
        arr = np.asarray(list(default), dtype=float)
    elif isinstance(value, str):
        arr = np.asarray([float(part.strip()) for part in value.split(",")], dtype=float)
    else:
        arr = np.asarray(list(value), dtype=float)
    if arr.shape != (6,):
        raise ValueError("Expected six comma-separated values")
    return arr


def clip_columns_by_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    cap = np.nanpercentile(values, percentile, axis=0)
    return np.minimum(values, cap[None, :])


def torque_smooth_deadband_observer(
    torque_error: np.ndarray,
    time: np.ndarray,
    observer_gain: float = 15.0,
    deadband_multiplier: float = 20.0,
    joint_deadband_multipliers: str | Iterable[float] | None = None,
    smooth_width_multiplier: float = 0.8,
) -> np.ndarray:
    """Smooth per-joint torque-space deadband before the residual observer.

    This suppresses small torque residuals without subtracting a hard offset.
    """
    joint_scale = parse_six_vector(
        joint_deadband_multipliers,
        default=(1.0, 1.0, 0.25, 1.0, 1.0, 1.0),
    )
    base_deadband = deadband_multiplier * joint_scale * available_signal_sigma(torque_error)
    corrected = smooth_deadband(
        torque_error,
        base_deadband,
        width=smooth_width_multiplier * base_deadband,
    )
    return residual_observer(corrected, time, observer_gain)


def torque_smooth_deadband_kalman(
    torque_error: np.ndarray,
    time: np.ndarray,
    deadband_multiplier: float = 20.0,
    joint_deadband_multipliers: str | Iterable[float] | None = None,
    smooth_width_multiplier: float = 0.8,
    kalman_process_scale: float = 0.08,
    kalman_measurement_scale: float = 1.0,
) -> np.ndarray:
    """Smooth per-joint torque-space deadband followed by a Kalman filter."""
    joint_scale = parse_six_vector(
        joint_deadband_multipliers,
        default=(1.0, 1.0, 0.25, 1.0, 1.0, 1.0),
    )
    base_deadband = deadband_multiplier * joint_scale * available_signal_sigma(torque_error)
    corrected = smooth_deadband(
        torque_error,
        base_deadband,
        width=smooth_width_multiplier * base_deadband,
    )
    return kalman_torque_filter(
        corrected,
        time,
        process_scale=kalman_process_scale,
        measurement_scale=kalman_measurement_scale,
    )


def kalman_torque_filter(
    torque_error: np.ndarray,
    time: np.ndarray,
    process_scale: float = 0.08,
    measurement_scale: float = 1.0,
    initial_variance_scale: float = 25.0,
) -> np.ndarray:
    """Realtime random-walk Kalman filter for external joint torque.

    State:       tau_ext,k = tau_ext,k-1 + w_k
    Measurement: y_k = tau_measured - tau_LSTM = tau_ext,k + v_k

    The six joints are filtered independently. Noise levels are estimated from
    the available torque residual signal only; no force sensor data is used.
    """
    if process_scale < 0.0 or measurement_scale <= 0.0 or initial_variance_scale <= 0.0:
        raise ValueError("Kalman scales must satisfy process >= 0 and measurement/initial > 0")

    measurement_sigma = measurement_scale * available_signal_sigma(torque_error)
    measurement_variance = np.maximum(measurement_sigma**2, 1e-18)
    process_variance = (process_scale * measurement_sigma) ** 2

    filtered = np.empty_like(torque_error)
    state = torque_error[:, 0].copy()
    covariance = initial_variance_scale * measurement_variance
    filtered[:, 0] = state

    dt = np.maximum(np.diff(time[: torque_error.shape[1]]), 0.0)
    median_dt = max(float(np.nanmedian(dt)), 1e-9) if dt.size else 1.0
    for idx, sample_dt in enumerate(dt, start=1):
        covariance = covariance + process_variance * max(sample_dt / median_dt, 1e-9)
        gain = covariance / (covariance + measurement_variance)
        state = state + gain * (torque_error[:, idx] - state)
        covariance = (1.0 - gain) * covariance
        filtered[:, idx] = state

    return filtered


def bias_gated_observer(
    torque_error: np.ndarray,
    time: np.ndarray,
    observer_gain: float = 15.0,
    bias_gain: float = 0.0,
    deadband_multiplier: float = 3.0,
) -> np.ndarray:
    """Observer with a slow free-space bias estimate and torque deadband.

    The bias is updated only while each channel looks close to its current
    free-space level. The observer then follows the bias-corrected residual
    after a soft deadband, reducing false positives from static model bias.
    """
    deadband = deadband_multiplier * available_signal_sigma(torque_error)
    bias = torque_error[:, 0].copy()
    corrected = np.empty_like(torque_error)
    corrected[:, 0] = soft_deadband((torque_error[:, [0]] - bias[:, None]), deadband)[:, 0]
    dt = np.maximum(np.diff(time[: torque_error.shape[1]]), 0.0)

    for idx, sample_dt in enumerate(dt, start=1):
        innovation = torque_error[:, idx] - bias
        free_space_like = np.abs(innovation) <= deadband
        bias_alpha = np.exp(-bias_gain * sample_dt)
        bias[free_space_like] = (
            bias_alpha * bias[free_space_like]
            + (1.0 - bias_alpha) * torque_error[free_space_like, idx]
        )
        corrected[:, idx] = soft_deadband((torque_error[:, [idx]] - bias[:, None]), deadband)[:, 0]

    return residual_observer(corrected, time, observer_gain)


def two_time_scale_observer(
    torque_error: np.ndarray,
    time: np.ndarray,
    observer_gain: float = 15.0,
    bias_gain: float = 0.0,
    deadband_multiplier: float = 3.0,
) -> np.ndarray:
    """Fast residual observer minus a gated slow bias observer."""
    fast = residual_observer(torque_error, time, observer_gain)
    deadband = deadband_multiplier * available_signal_sigma(torque_error)
    slow_bias = np.empty_like(torque_error)
    slow_bias[:, 0] = torque_error[:, 0]
    dt = np.maximum(np.diff(time[: torque_error.shape[1]]), 0.0)

    for idx, sample_dt in enumerate(dt, start=1):
        innovation = fast[:, idx] - slow_bias[:, idx - 1]
        free_space_like = np.abs(innovation) <= deadband
        alpha = np.exp(-bias_gain * sample_dt)
        slow_bias[:, idx] = slow_bias[:, idx - 1]
        slow_bias[free_space_like, idx] = (
            alpha * slow_bias[free_space_like, idx - 1]
            + (1.0 - alpha) * fast[free_space_like, idx]
        )

    return soft_deadband(fast - slow_bias, deadband)


def external_torque_estimate(
    joint_data: np.ndarray,
    torque_data: np.ndarray,
    estimator_mode: str = "observer",
    observer_gain: float = 15.0,
    bias_gain: float = 0.0,
    deadband_multiplier: float = 3.0,
    joint_deadband_multipliers: str | Iterable[float] | None = None,
    smooth_width_multiplier: float = 0.8,
    kalman_process_scale: float = 0.08,
    kalman_measurement_scale: float = 1.0,
) -> np.ndarray:
    """Return direct or observer-filtered external joint torque estimate."""
    torque_error = torque_error_from_data(joint_data, torque_data)
    if estimator_mode == "direct":
        return torque_error
    time = joint_data[: torque_error.shape[1], 0]
    if estimator_mode == "observer":
        return residual_observer(torque_error, time, observer_gain)
    if estimator_mode == "kalman":
        return kalman_torque_filter(
            torque_error,
            time,
            process_scale=kalman_process_scale,
            measurement_scale=kalman_measurement_scale,
        )
    if estimator_mode == "kalman_smooth_deadband":
        return torque_smooth_deadband_kalman(
            torque_error,
            time,
            deadband_multiplier=deadband_multiplier,
            joint_deadband_multipliers=joint_deadband_multipliers,
            smooth_width_multiplier=smooth_width_multiplier,
            kalman_process_scale=kalman_process_scale,
            kalman_measurement_scale=kalman_measurement_scale,
        )
    if estimator_mode == "bias_gate":
        return bias_gated_observer(
            torque_error,
            time,
            observer_gain=observer_gain,
            bias_gain=bias_gain,
            deadband_multiplier=deadband_multiplier,
        )
    if estimator_mode == "two_time_scale":
        return two_time_scale_observer(
            torque_error,
            time,
            observer_gain=observer_gain,
            bias_gain=bias_gain,
            deadband_multiplier=deadband_multiplier,
        )
    if estimator_mode == "torque_smooth_deadband":
        return torque_smooth_deadband_observer(
            torque_error,
            time,
            observer_gain=observer_gain,
            deadband_multiplier=deadband_multiplier,
            joint_deadband_multipliers=joint_deadband_multipliers,
            smooth_width_multiplier=smooth_width_multiplier,
        )
    raise ValueError(
        "estimator_mode must be 'direct', 'observer', 'kalman', "
        "'kalman_smooth_deadband', 'bias_gate', 'two_time_scale', "
        "or 'torque_smooth_deadband'"
    )


def estimate_external_force(
    joint_data: np.ndarray,
    jacobian_data: np.ndarray,
    torque_data: np.ndarray,
    estimator_mode: str = "observer",
    observer_gain: float = 15.0,
    bias_gain: float = 0.0,
    deadband_multiplier: float = 3.0,
    joint_deadband_multipliers: str | Iterable[float] | None = None,
    smooth_width_multiplier: float = 0.8,
    kalman_process_scale: float = 0.08,
    kalman_measurement_scale: float = 1.0,
) -> np.ndarray:
    n_samples = torque_data.shape[0]
    jacobian = build_jacobian(jacobian_data, n_samples)
    external_torque = external_torque_estimate(
        joint_data,
        torque_data,
        estimator_mode,
        observer_gain,
        bias_gain,
        deadband_multiplier,
        joint_deadband_multipliers,
        smooth_width_multiplier,
        kalman_process_scale,
        kalman_measurement_scale,
    )
    force_rotation = force_rotation_matrix()

    force = np.empty((6, n_samples), dtype=float)
    for idx in range(n_samples):
        force[:, idx] = np.linalg.solve(jacobian[:, :, idx].T, external_torque[:, idx])
        force[:3, idx] = force_rotation @ force[:3, idx]
    return force.T


def robust_sigma(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.full(6, np.nan)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    sigma = 1.4826 * mad
    rmse = np.sqrt(np.mean(values**2, axis=0))
    return np.maximum(sigma, 0.35 * rmse)


def monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """Force sigma to be non-increasing as estimated contact magnitude increases."""
    out = values.copy()
    for col in range(out.shape[1]):
        valid = np.isfinite(out[:, col])
        if not np.any(valid):
            continue
        series = out[valid, col]
        out[valid, col] = np.maximum.accumulate(series[::-1])[::-1]
    return out


def estimate_joint_velocity(joint_data: np.ndarray, n_samples: int) -> np.ndarray:
    """Use recorded joint velocity columns, falling back to finite differences."""
    if joint_data.shape[1] >= 13:
        velocity = joint_data[:n_samples, 7:13]
        if np.nanmax(np.abs(velocity)) > 0:
            return velocity

    time = joint_data[:n_samples, 0]
    position = joint_data[:n_samples, 1:7]
    return np.gradient(position, time, axis=0)


def estimate_cartesian_velocity(
    joint_data: np.ndarray,
    jacobian_data: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate tool twist from J qdot and return rotated Cartesian speed."""
    joint_velocity = estimate_joint_velocity(joint_data, n_samples)
    jacobian = build_jacobian(jacobian_data, n_samples)
    force_rotation = force_rotation_matrix()

    cartesian_velocity = np.empty_like(joint_velocity)
    for idx in range(n_samples):
        cartesian_velocity[idx] = jacobian[:, :, idx] @ joint_velocity[idx]
        cartesian_velocity[idx, :3] = force_rotation @ cartesian_velocity[idx, :3]
    cartesian_speed = np.linalg.norm(cartesian_velocity[:, :3], axis=1)
    return cartesian_velocity, cartesian_speed


def available_signal_sigma(torque_error: np.ndarray) -> np.ndarray:
    """Estimate torque-side noise from available signals only.

    The first difference suppresses slow contact-force content and leaves a
    conservative estimate of torque prediction jitter/friction fluctuation.
    No force-sensor values are used here.
    """
    delta = np.diff(torque_error, axis=1)
    sigma = robust_sigma(delta.T) / np.sqrt(2.0)
    floor = 1e-6 * np.maximum(1.0, np.median(np.abs(torque_error), axis=1))
    return np.maximum(sigma, floor)


def leak_free_contact_dependent_ci(
    estimate: np.ndarray,
    external_torque: np.ndarray,
    joint_data: np.ndarray,
    jacobian_data: np.ndarray,
    torque_data: np.ndarray,
    n_bins: int = 12,
    confidence: float = 0.95,
    low_force_gain: float = 3.0,
    force_scale: float | None = None,
    observer_disagreement_gain: float = 0.8,
    relative_force_uncertainty: float = 0.10,
    fz_uncertainty_gain: float = 2.0,
    max_ci_half_width: float | None = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return leak-free lower, upper, half-width, bin centers, and binned sigma.

    The force sensor is not used. The band is formed by propagating torque-side
    uncertainty through J^-T, then adding a model-risk term when the realtime
    observer disagrees with the instantaneous torque projection.
    """
    n_samples = estimate.shape[0]
    jacobian = build_jacobian(jacobian_data, n_samples)
    raw_torque_error = torque_error_from_data(joint_data, torque_data)[:, :n_samples]
    base_tau_sigma = available_signal_sigma(raw_torque_error)
    raw_torque_error = torque_error_from_data(joint_data, torque_data)[:, :n_samples].T

    force_magnitude = np.linalg.norm(estimate[:, :3], axis=1)

    if force_scale is None:
        force_scale = 10.0

    low_force_factor = 1.0 + low_force_gain * np.exp(-force_magnitude / force_scale)

    rot6 = np.eye(6)
    rot6[:3, :3] = force_rotation_matrix()
    sigma = np.empty_like(estimate)
    observer_disagreement_sigma = np.empty_like(estimate)
    for idx in range(n_samples):
        transform = rot6 @ np.linalg.inv(jacobian[:, :, idx].T)
        channel_variance = (transform**2) @ ((base_tau_sigma * low_force_factor[idx]) ** 2)
        direct_force = transform @ raw_torque_error[idx]
        propagated_sigma = np.sqrt(np.maximum(channel_variance, 0.0))
        observer_disagreement_sigma[idx] = np.abs(direct_force - estimate[idx])
        sigma[idx] = np.sqrt(
            propagated_sigma**2
            + (observer_disagreement_gain * observer_disagreement_sigma[idx]) ** 2
        )

    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-12 else normal_z(confidence)
    half_width = z * sigma
    half_width[:, :3] = np.sqrt(
        half_width[:, :3] ** 2
        + (relative_force_uncertainty * np.abs(estimate[:, :3])) ** 2
    )
    half_width[:, 2] *= fz_uncertainty_gain
    if max_ci_half_width is not None:
        half_width = np.minimum(half_width, max_ci_half_width)

    max_mag = float(np.nanmax(force_magnitude))
    edges = np.quantile(force_magnitude, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([0.0, max_mag + 1e-12])
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned_sigma = np.full((centers.size, estimate.shape[1]), np.nan)
    for idx in range(centers.size):
        mask = (force_magnitude >= edges[idx]) & (force_magnitude < edges[idx + 1])
        if idx == centers.size - 1:
            mask = (force_magnitude >= edges[idx]) & (force_magnitude <= edges[idx + 1])
        if np.count_nonzero(mask) > 0:
            binned_sigma[idx] = np.median(half_width[mask] / z, axis=0)
    for col in range(estimate.shape[1]):
        valid = np.isfinite(binned_sigma[:, col])
        if np.count_nonzero(valid) == 0:
            binned_sigma[:, col] = np.median(half_width[:, col] / z)
        else:
            binned_sigma[:, col] = np.interp(centers, centers[valid], binned_sigma[valid, col])

    return estimate - half_width, estimate + half_width, half_width, centers, binned_sigma


def normal_z(confidence: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def run_pipeline(
    data_root: Path,
    data: str = "free_space",
    rnn: str = "lstm",
    network: str = "_seal_pred_filtered_torque_si_9_1.csv",
    window_size: int = 1,
    estimator_mode: str = "observer",
    observer_gain: float = 15.0,
    bias_gain: float = 0.0,
    deadband_multiplier: float = 3.0,
    joint_deadband_multipliers: str | Iterable[float] | None = None,
    smooth_width_multiplier: float = 0.8,
    kalman_process_scale: float = 0.08,
    kalman_measurement_scale: float = 1.0,
    n_bins: int = 12,
    confidence: float = 0.95,
    low_force_gain: float = 3.0,
    force_scale: float | None = None,
    observer_disagreement_gain: float = 0.8,
    relative_force_uncertainty: float = 0.10,
    fz_uncertainty_gain: float = 2.0,
    max_ci_half_width: float | None = 12.0,
) -> ForceResults:
    run_root = data_root / "test" / data
    joint_data = read_matrix(run_root / "joints" / "interpolated_all_joints.csv")
    jacobian_data = read_matrix(run_root / "jacobian" / "interpolated_all_jacobian.csv")
    torque_data = read_matrix(run_root / f"{rnn}{network}")
    real_force_data = read_matrix(data_root / "test" / "sensor" / "interpolated_all_sensor.csv")

    external_torque = external_torque_estimate(
        joint_data,
        torque_data,
        estimator_mode=estimator_mode,
        observer_gain=observer_gain,
        bias_gain=bias_gain,
        deadband_multiplier=deadband_multiplier,
        joint_deadband_multipliers=joint_deadband_multipliers,
        smooth_width_multiplier=smooth_width_multiplier,
        kalman_process_scale=kalman_process_scale,
        kalman_measurement_scale=kalman_measurement_scale,
    ).T
    raw_force = estimate_external_force(
        joint_data,
        jacobian_data,
        torque_data,
        estimator_mode=estimator_mode,
        observer_gain=observer_gain,
        bias_gain=bias_gain,
        deadband_multiplier=deadband_multiplier,
        joint_deadband_multipliers=joint_deadband_multipliers,
        smooth_width_multiplier=smooth_width_multiplier,
        kalman_process_scale=kalman_process_scale,
        kalman_measurement_scale=kalman_measurement_scale,
    )
    force = moving_average(raw_force, window_size)
    real_force = real_force_data[: force.shape[0], 1:7]
    time = joint_data[: force.shape[0], 0]
    joint_velocity = estimate_joint_velocity(joint_data, force.shape[0])
    cartesian_velocity, cartesian_speed = estimate_cartesian_velocity(
        joint_data,
        jacobian_data,
        force.shape[0],
    )
    rmse = np.sqrt(np.mean((force - real_force) ** 2, axis=0))

    ci_lower, ci_upper, ci_half_width, centers, sigma = leak_free_contact_dependent_ci(
        force,
        external_torque,
        joint_data,
        jacobian_data,
        torque_data,
        n_bins=n_bins,
        confidence=confidence,
        low_force_gain=low_force_gain,
        force_scale=force_scale,
        observer_disagreement_gain=observer_disagreement_gain,
        relative_force_uncertainty=relative_force_uncertainty,
        fz_uncertainty_gain=fz_uncertainty_gain,
        max_ci_half_width=max_ci_half_width,
    )

    return ForceResults(
        time=time,
        external_torque=external_torque,
        raw_force=raw_force,
        force=force,
        real_force=real_force,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_half_width=ci_half_width,
        rmse=rmse,
        sigma_bin_centers=centers,
        sigma_by_channel=sigma,
        joint_velocity=joint_velocity,
        cartesian_velocity=cartesian_velocity,
        cartesian_speed=cartesian_speed,
        estimator_mode=estimator_mode,
        observer_gain=observer_gain,
        bias_gain=bias_gain,
        deadband_multiplier=deadband_multiplier,
    )


def save_results_csv(results: ForceResults, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    header = ["time"]
    arrays: list[np.ndarray] = [results.time[:, None]]
    for name, arr in (
        ("external_torque", results.external_torque),
        ("truth", results.real_force),
        ("estimate", results.force),
        ("ci_lower", results.ci_lower),
        ("ci_upper", results.ci_upper),
        ("ci_half_width", results.ci_half_width),
    ):
        header.extend(f"{name}_{channel}" for channel in CHANNELS)
        arrays.append(arr)
    header.extend(f"joint_velocity_{channel}" for channel in JOINT_VELOCITY_CHANNELS)
    arrays.append(results.joint_velocity)
    header.extend(f"cartesian_velocity_{channel}" for channel in CARTESIAN_VELOCITY_CHANNELS)
    arrays.append(results.cartesian_velocity)
    header.append("cartesian_speed")
    arrays.append(results.cartesian_speed[:, None])
    output_path = output_dir / "external_force_with_ci.csv"
    np.savetxt(
        output_path,
        np.column_stack(arrays),
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    return output_path


def plot_force_ci(
    results: ForceResults,
    output_dir: Path,
    channels: Iterable[int] = (0, 1, 2),
    highlight_regions: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_list = list(channels)
    fig, axes = plt.subplots(len(channel_list), 1, figsize=(12, 8), sharex=True)
    if len(channel_list) == 1:
        axes = [axes]

    for ax, channel in zip(axes, channel_list):
        if highlight_regions:
            ax.axvspan(18.0, 28.0, color="0.1", alpha=0.06, linewidth=0)
            ax.axvspan(52.0, 56.5, color="0.1", alpha=0.06, linewidth=0)
            ax.axvspan(112.0, 123.0, color="tab:orange", alpha=0.10, linewidth=0)
        ax.fill_between(
            results.time,
            results.ci_lower[:, channel],
            results.ci_upper[:, channel],
            color="tab:red",
            alpha=0.20,
            linewidth=0,
            label="95% interval",
        )
        ax.plot(results.time, results.real_force[:, channel], color="tab:blue", linewidth=1.0, label="measured")
        ax.plot(results.time, results.force[:, channel], color="tab:red", linewidth=1.0, label="estimated")
        ax.set_ylabel("Force (N)" if channel < 3 else "Torque (Nm)")
        ax.set_title(f"{CHANNELS[channel]}, RMSE = {results.rmse[channel]:.4f}")
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    fig.suptitle(
        f"External Force Estimation ({results.estimator_mode}, K_i={results.observer_gain:g}) "
        "with Contact-Dependent 95% Confidence Bands"
    )
    fig.tight_layout()
    output_path = output_dir / "external_force_ci.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_sigma_model(results: ForceResults, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for channel in range(3):
        ax.plot(
            results.sigma_bin_centers,
            1.959963984540054 * results.sigma_by_channel[:, channel],
            marker="o",
            linestyle="none",
            label=f"{CHANNELS[channel]} half-width model",
        )
    ax.set_xlabel("Estimated contact-force magnitude (N)")
    ax.set_ylabel("Median 95% half-width (N)")
    ax.set_title("Binned confidence-band width from the leak-free uncertainty model")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / "ci_width_vs_contact_force.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_velocity_uncertainty_context(
    results: ForceResults,
    output_dir: Path,
    highlight_regions: bool = True,
    smoothing_window: int = 101,
) -> Path:
    """Plot tool speed and force CI half-widths on the same time base."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    speed = centered_moving_average(results.cartesian_speed[:, None], smoothing_window)[:, 0]
    half_width = centered_moving_average(results.ci_half_width[:, :3], smoothing_window)

    if highlight_regions:
        for ax in axes:
            ax.axvspan(18.0, 28.0, color="0.1", alpha=0.06, linewidth=0)
            ax.axvspan(52.0, 56.5, color="0.1", alpha=0.06, linewidth=0)
            ax.axvspan(112.0, 123.0, color="tab:orange", alpha=0.10, linewidth=0)

    axes[0].plot(results.time, speed, color="0.15", linewidth=1.2)
    axes[0].set_ylabel("Tool speed")
    axes[0].set_title("Velocity Context for Force-Uncertainty Band")
    axes[0].grid(True, alpha=0.35)

    for channel in range(3):
        axes[1].plot(
            results.time,
            half_width[:, channel],
            linewidth=1.2,
            label=f"{CHANNELS[channel]} CI half-width",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Smoothed force CI half-width (N)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    fig.tight_layout()
    output_path = output_dir / "velocity_with_force_uncertainty.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_ci_width_vs_velocity(results: ForceResults, output_dir: Path, n_bins: int = 12) -> Path:
    """Plot binned force CI half-width against tool speed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    speed = results.cartesian_speed
    edges = np.quantile(speed, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([float(np.nanmin(speed)), float(np.nanmax(speed)) + 1e-12])
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned_half_width = np.full((centers.size, 3), np.nan)

    for idx in range(centers.size):
        mask = (speed >= edges[idx]) & (speed < edges[idx + 1])
        if idx == centers.size - 1:
            mask = (speed >= edges[idx]) & (speed <= edges[idx + 1])
        if np.count_nonzero(mask) > 0:
            binned_half_width[idx] = np.median(results.ci_half_width[mask, :3], axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    for channel in range(3):
        ax.plot(
            centers,
            binned_half_width[:, channel],
            marker="o",
            linestyle="none",
            label=f"{CHANNELS[channel]} half-width",
        )
    ax.set_xlabel("Tool speed from J qdot")
    ax.set_ylabel("Median 95% force CI half-width (N)")
    ax.set_title("Force-Uncertainty Width Versus Tool Speed")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / "ci_width_vs_velocity.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--data", default="free_space")
    parser.add_argument("--rnn", default="lstm")
    parser.add_argument("--network", default="_seal_pred_filtered_torque_si_9_1.csv")
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument(
        "--estimator",
        choices=(
            "observer",
            "direct",
            "kalman",
            "kalman_smooth_deadband",
            "bias_gate",
            "two_time_scale",
            "torque_smooth_deadband",
        ),
        default="observer",
    )
    parser.add_argument("--observer-gain", type=float, default=15.0)
    parser.add_argument("--bias-gain", type=float, default=0.0)
    parser.add_argument("--deadband-multiplier", type=float, default=3.0)
    parser.add_argument("--joint-deadband-multipliers", default=None)
    parser.add_argument("--smooth-width-multiplier", type=float, default=0.8)
    parser.add_argument("--kalman-process-scale", type=float, default=0.08)
    parser.add_argument("--kalman-measurement-scale", type=float, default=1.0)
    parser.add_argument("--ci-bins", type=int, default=12)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--low-force-gain", type=float, default=3.0)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--observer-disagreement-gain", type=float, default=0.8)
    parser.add_argument("--relative-force-uncertainty", type=float, default=0.10)
    parser.add_argument("--fz-uncertainty-gain", type=float, default=2.0)
    parser.add_argument("--max-ci-half-width", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        data_root=args.data_root,
        data=args.data,
        rnn=args.rnn,
        network=args.network,
        window_size=args.window_size,
        estimator_mode=args.estimator,
        observer_gain=args.observer_gain,
        bias_gain=args.bias_gain,
        deadband_multiplier=args.deadband_multiplier,
        joint_deadband_multipliers=args.joint_deadband_multipliers,
        smooth_width_multiplier=args.smooth_width_multiplier,
        kalman_process_scale=args.kalman_process_scale,
        kalman_measurement_scale=args.kalman_measurement_scale,
        n_bins=args.ci_bins,
        confidence=args.confidence,
        low_force_gain=args.low_force_gain,
        force_scale=args.force_scale,
        observer_disagreement_gain=args.observer_disagreement_gain,
        relative_force_uncertainty=args.relative_force_uncertainty,
        fz_uncertainty_gain=args.fz_uncertainty_gain,
        max_ci_half_width=args.max_ci_half_width,
    )

    csv_path = save_results_csv(results, args.output_dir)
    print(f"Saved samples with confidence intervals: {csv_path}")
    if not args.no_plots:
        force_plot = plot_force_ci(results, args.output_dir)
        sigma_plot = plot_sigma_model(results, args.output_dir)
        velocity_context_plot = plot_velocity_uncertainty_context(results, args.output_dir)
        velocity_width_plot = plot_ci_width_vs_velocity(results, args.output_dir, n_bins=args.ci_bins)
        print(f"Saved force plot: {force_plot}")
        print(f"Saved CI-width plot: {sigma_plot}")
        print(f"Saved velocity context plot: {velocity_context_plot}")
        print(f"Saved CI-vs-velocity plot: {velocity_width_plot}")

    print("RMSE:")
    print(
        f"Estimator: {results.estimator_mode}, observer_gain: {results.observer_gain:g} 1/s, "
        f"bias_gain: {results.bias_gain:g} 1/s, deadband_multiplier: {results.deadband_multiplier:g}"
    )
    for channel, value in zip(CHANNELS, results.rmse):
        print(f"  {channel}: {value:.6f}")

    estimated_magnitude = np.linalg.norm(results.force[:, :3], axis=1)
    low_width = 2.0 * results.ci_half_width[estimated_magnitude.argmin(), :3]
    high_width = 2.0 * results.ci_half_width[estimated_magnitude.argmax(), :3]
    print("Approx. full 95% band width at smallest vs largest estimated force magnitude:")
    for idx, channel in enumerate(CHANNELS[:3]):
        print(f"  {channel}: near zero {low_width[idx]:.3f} N, large force {high_width[idx]:.3f} N")


if __name__ == "__main__":
    main()
