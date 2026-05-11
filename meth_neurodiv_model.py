"""
Computational Model of Methamphetamine Perturbations in Neurodivergent
Reward and Arousal Networks

Multi-scale simulation integrating:
  Layer 1: Dopamine terminal kinetics (ms-min)
  Layer 2: Temperature-dependent oxidative toxicity (min-hours)
  Layer 3: LC-NE gain control (seconds)
  Layer 4: Reinforcement learning (trial-level)
  Layer 5: Sleep/circadian modulation (hours)
  Layer 6: Chronic neuroadaptation (days-months)

Author: Evelyn Campbell
DOI: 10.5281/ZENODO.19625787
Date: March 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import os

# ─────────────────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────────────────
COLORS = {
    'bg': '#0a0a0a', 'amber': '#e8a838', 'purple': '#9b59b6',
    'teal': '#1abc9c', 'red': '#e74c3c', 'blue': '#3498db',
    'white': '#ecf0f1', 'gray': '#7f8c8d', 'green': '#2ecc71',
    'orange': '#e67e22',
}

def setup_plot_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
        'axes.edgecolor': COLORS['gray'], 'axes.labelcolor': COLORS['white'],
        'text.color': COLORS['white'], 'xtick.color': COLORS['white'],
        'ytick.color': COLORS['white'], 'legend.facecolor': '#1a1a1a',
        'legend.edgecolor': COLORS['gray'], 'legend.labelcolor': COLORS['white'],
        'font.size': 10, 'axes.titlesize': 12, 'figure.dpi': 150,
        'savefig.facecolor': COLORS['bg'], 'savefig.edgecolor': COLORS['bg'],
    })

setup_plot_style()

# ─────────────────────────────────────────────────────────
# PARAMETER STRUCTURES
# ─────────────────────────────────────────────────────────

@dataclass
class TerminalParams:
    """Dopamine terminal kinetics parameters."""
    q: float = 3000.0       # vesicular quantum (molecules)
    p: float = 0.06         # release probability
    F_tonic: float = 4.0    # tonic firing rate (Hz)
    Km: float = 0.16        # DAT Michaelis constant (µM)
    Vmax: float = 4.0       # DAT max velocity (µM/s)
    k_diff: float = 0.5     # diffusion clearance (s⁻¹)
    k_rev: float = 8.0      # reverse transport rate (s⁻¹)
    K_AMPH: float = 0.75    # Hill K for amphetamine reversal (µM)
    n_hill: float = 2.0     # Hill coefficient
    k_leak: float = 0.01    # vesicular leak (s⁻¹)
    k_MAO: float = 0.1      # MAO metabolism (s⁻¹)
    k_pack: float = 0.5     # VMAT2 packing rate (s⁻¹)
    De_baseline: float = 0.025   # baseline extracellular DA (µM)
    Dc_baseline: float = 1.0     # baseline cytoplasmic DA (µM)
    Dv_baseline: float = 100.0   # baseline vesicular DA (normalized)
    # Attached by simulate_acute_da_response for ODE access
    k0_ox: float = 0.001
    Q10: float = 2.5
    k_deg: float = 0.0001
    gamma_pdat: float = 0.5

@dataclass
class ToxicityParams:
    """Temperature-dependent neurotoxicity parameters."""
    k0_ox: float = 0.001       # baseline oxidation rate at 37°C (s⁻¹)
    Q10: float = 2.5           # temperature coefficient
    T_baseline: float = 37.0   # baseline body temperature (°C)
    k_deg: float = 0.0001      # terminal damage rate
    T_meth_rise: float = 3.0   # max temperature rise under METH (°C)
    tau_temp: float = 1800.0   # temperature rise time constant (s)

@dataclass
class LCParams:
    """Locus coeruleus / norepinephrine parameters."""
    I_exc: float = 2.2     # LC excitatory drive
    g_alpha2: float = 1.0  # α2 autoreceptor gain
    tau_L: float = 5.0     # LC time constant (s)
    G0: float = 1.0        # baseline gain
    alpha_G: float = 0.3   # gain-LC slope
    NE_baseline: float = 1.0  # baseline NE level

@dataclass
class RLParams:
    """Reinforcement learning parameters (TD-learning layer)."""
    alpha0: float = 0.10    # baseline learning rate
    k1_alpha: float = 0.06  # RPE+ modulation of learning rate
    k2_alpha: float = 0.02  # RPE- modulation
    gamma: float = 0.95     # temporal discount
    beta: float = 5.0       # inverse temperature (softmax)
    w0: float = 0.5         # initial value weight

@dataclass
class SleepParams:
    """Sleep/circadian modulation parameters (Process S)."""
    tau_w: float = 18.2 * 3600  # wake time constant (s)
    tau_s: float = 4.2 * 3600   # sleep time constant (s)
    k_sleep: float = 0.04 / 60  # DAT phosphorylation rate (s⁻¹)
    k_wake: float = 0.03 / 60   # dephosphorylation rate (s⁻¹)
    gamma_pdat: float = 0.5     # Vmax enhancement from phosphorylation
    k_SD_D2: float = 0.001      # sleep deprivation → D2 downregulation

@dataclass
class ChronicParams:
    """Long-timescale neuroadaptation parameters."""
    k_DAT_up: float = 2.74e-10  # DAT upregulation rate (~24%/yr)
    k_DAT_decay: float = 1e-8   # DAT return to baseline
    lambda_allo: float = 1e-6   # allostatic setpoint drift rate

@dataclass
class NeuralArchitecture:
    """Complete parameter set for a neural architecture."""
    name: str = "Neurotypical"
    terminal: TerminalParams = field(default_factory=TerminalParams)
    toxicity: ToxicityParams = field(default_factory=ToxicityParams)
    lc: LCParams = field(default_factory=LCParams)
    rl: RLParams = field(default_factory=RLParams)
    sleep: SleepParams = field(default_factory=SleepParams)
    chronic: ChronicParams = field(default_factory=ChronicParams)


# ─────────────────────────────────────────────────────────
# ARCHITECTURE CONSTRUCTORS
# ─────────────────────────────────────────────────────────

def make_neurotypical() -> NeuralArchitecture:
    return NeuralArchitecture(name="Neurotypical")

def make_adhd_c() -> NeuralArchitecture:
    """ADHD-Combined: low tonic DA, high phasic, elevated LC excitability."""
    arch = NeuralArchitecture(name="ADHD-C")
    arch.terminal.Vmax *= 0.85       # ~15% lower DAT capacity
    arch.terminal.F_tonic = 3.5      # slightly reduced tonic firing
    arch.lc.I_exc = 3.5              # elevated LC excitability
    arch.lc.g_alpha2 = 0.55          # impaired α2 autoreceptor feedback
    arch.rl.alpha0 = 0.08            # slightly lower baseline LR
    arch.rl.k1_alpha = 0.08          # amplified positive RPE
    arch.rl.beta = 3.5               # reduced action selectivity
    return arch

def make_adhd_i() -> NeuralArchitecture:
    """ADHD-Inattentive: NE-dominant deficit, mild DA changes (Hypothesis A)."""
    arch = NeuralArchitecture(name="ADHD-I")
    arch.terminal.Vmax *= 0.90       # modest DAT reduction
    arch.lc.I_exc = 1.2              # LC hypoarousal
    arch.lc.g_alpha2 = 1.0           # normal α2
    arch.rl.alpha0 = 0.09
    arch.rl.beta = 4.0               # somewhat reduced selectivity
    return arch


# ─────────────────────────────────────────────────────────
# LAYER 1+2: DOPAMINE TERMINAL KINETICS + TOXICITY
# ─────────────────────────────────────────────────────────

def da_terminal_odes(t, y, params: TerminalParams, drug_conc: float,
                     drug_type: str, T: float, P_DAT: float = 0.0):
    """
    Fast-layer ODEs for dopamine terminal dynamics.

    State variables:
      y[0] = De  — extracellular DA (µM)
      y[1] = Dc  — cytoplasmic DA (µM)
      y[2] = Dv  — vesicular DA (normalized)
      y[3] = Q   — quinone accumulation
      y[4] = N   — terminal integrity (0–1)
    """
    De, Dc, Dv, Q, N = y
    p = params

    De = max(De, 0.0)
    Dc = max(Dc, 0.0)
    Dv = max(Dv, 0.0)
    N  = float(np.clip(N, 0.0, 1.0))

    # Vesicular release (scaled)
    R = p.q * p.p * p.F_tonic * Dv / p.Dv_baseline * N * 1e-6

    # Phosphorylation-modulated Vmax (Sleep Layer coupling)
    Vmax_eff = p.Vmax * (1.0 + p.gamma_pdat * P_DAT)

    # Drug-specific transport modes
    if drug_type == 'MPH':
        Km_eff = p.Km * (1.0 + drug_conc / 0.1)  # competitive inhibition
        U_DAT = Vmax_eff * De / (Km_eff + De)
        E_DAT = 0.0
        k_pack_eff = p.k_pack

    elif drug_type in ('AMPH', 'METH'):
        P_rev = drug_conc**p.n_hill / (p.K_AMPH**p.n_hill + drug_conc**p.n_hill)
        E_DAT = p.k_rev * P_rev * Dc * N
        U_DAT = Vmax_eff * De / (p.Km + De) * (1.0 - P_rev)
        vmat2_block = 0.9 if drug_type == 'METH' else 0.5
        k_pack_eff = p.k_pack * (1.0 - vmat2_block * P_rev)

    else:  # no drug
        U_DAT = Vmax_eff * De / (p.Km + De)
        E_DAT = 0.0
        k_pack_eff = p.k_pack

    # Temperature-dependent oxidation (Q10 scaling)
    k_ox = p.k0_ox * p.Q10**((T - 37.0) / 10.0)

    # Temperature-gating for terminal damage (sigmoid threshold ~40°C)
    theta_T = 1.0 / (1.0 + np.exp(-(T - 40.0) / 0.5))

    dDe = R + E_DAT - U_DAT - p.k_diff * De
    dDc = p.k_leak * Dv - p.k_MAO * Dc - E_DAT + U_DAT * 0.1
    dDv = k_pack_eff * Dc - p.k_leak * Dv
    dQ  = k_ox * Dc * (T / 37.0)
    dN  = -p.k_deg * Q * theta_T

    return [dDe, dDc, dDv, dQ, dN]


# ─────────────────────────────────────────────────────────
# LAYER 3: LC-NE GAIN CONTROL
# ─────────────────────────────────────────────────────────

def lc_dynamics(L: float, NE: float, lc: LCParams,
                dt: float) -> Tuple[float, float, float]:
    """Update LC firing rate, NE level, and gain (Aston-Jones & Cohen model)."""
    dL = (lc.I_exc - lc.g_alpha2 * NE - L) / lc.tau_L
    L_new  = max(0.0, L + dL * dt)
    NE_new = 0.9 * NE + 0.1 * L_new          # smoothed NE tracking
    G = lc.G0 + lc.alpha_G * (L_new - 2.0)   # inverted-U gain
    G = float(np.clip(G, 0.3, 3.0))
    return L_new, NE_new, G


# ─────────────────────────────────────────────────────────
# LAYER 4: REINFORCEMENT LEARNING (TD)
# ─────────────────────────────────────────────────────────

def compute_prediction_error(reward: float, V_current: float,
                              V_next: float, gamma: float) -> float:
    return reward + gamma * V_next - V_current

def update_values(V: np.ndarray, state: int, delta: float,
                  alpha_plus: float, alpha_minus: float) -> np.ndarray:
    """Asymmetric learning update."""
    lr = alpha_plus if delta >= 0 else alpha_minus
    V[state] += lr * delta
    return V


# ─────────────────────────────────────────────────────────
# LAYER 5: SLEEP / CIRCADIAN MODULATION
# ─────────────────────────────────────────────────────────

def sleep_modulation(t_hours: float, sp: SleepParams,
                     sleep_deprived: bool = False) -> Tuple[float, float]:
    """
    Compute DAT phosphorylation state and Process-S homeostatic pressure.

    Returns
    -------
    P_DAT : float  — DAT phosphorylation fraction (0–1)
    S     : float  — sleep pressure (0–1)
    """
    # Circadian modulation: sine approximation
    phase = 2 * np.pi * t_hours / 24.0
    S = 0.5 + 0.5 * np.sin(phase - np.pi / 2)  # peaks ~midnight

    # Sleep deprivation raises phosphorylation (reduces DAT Vmax)
    if sleep_deprived:
        P_DAT = min(1.0, S + 0.3)
    else:
        P_DAT = S * 0.5  # partial phosphorylation during normal wakefulness

    return P_DAT, S


# ─────────────────────────────────────────────────────────
# LAYER 6: CHRONIC NEUROADAPTATION
# ─────────────────────────────────────────────────────────

def chronic_adaptation_step(DAT_scale: float, setpoint: float,
                             De_mean: float, cp: ChronicParams,
                             dt_days: float) -> Tuple[float, float]:
    """
    Single time-step of long-timescale neuroadaptation.

    Returns updated (DAT_scale, setpoint).
    """
    # DAT upregulation in response to excess extracellular DA
    dDAT = cp.k_DAT_up * (De_mean - 0.025) * dt_days * 86400 \
           - cp.k_DAT_decay * (DAT_scale - 1.0)
    DAT_scale += dDAT
    DAT_scale  = max(0.3, min(2.0, DAT_scale))

    # Allostatic setpoint drift
    d_set = cp.lambda_allo * (De_mean - setpoint) * dt_days * 86400
    setpoint += d_set

    return DAT_scale, setpoint


# ─────────────────────────────────────────────────────────
# SIMULATION WRAPPERS
# ─────────────────────────────────────────────────────────

def simulate_acute_da_response(arch: NeuralArchitecture,
                                drug_type: str = 'METH',
                                drug_conc: float = 5.0,
                                duration: float = 3600.0,
                                sleep_deprived: bool = False) -> Dict:
    """
    Simulate acute drug exposure (Layers 1+2).

    Parameters
    ----------
    arch       : NeuralArchitecture
    drug_type  : 'METH', 'AMPH', 'MPH', or 'none'
    drug_conc  : peak drug concentration (µM)
    duration   : simulation duration (seconds)
    sleep_deprived : whether DAT phosphorylation is elevated

    Returns dict with time-series of all state variables.
    """
    p   = arch.terminal
    tox = arch.toxicity

    # Attach toxicity scalars to terminal params for ODE access
    p.k0_ox    = tox.k0_ox
    p.Q10      = tox.Q10
    p.k_deg    = tox.k_deg
    p.gamma_pdat = arch.sleep.gamma_pdat

    y0 = [p.De_baseline, p.Dc_baseline, p.Dv_baseline, 0.0, 1.0]

    def drug_profile(t):
        if t < 60:
            return drug_conc * (1.0 - np.exp(-t / 15.0))
        elif t < duration * 0.7:
            return drug_conc * np.exp(-(t - 60.0) / (duration * 0.5))
        else:
            return drug_conc * 0.01

    def temp_profile(t):
        T_rise = tox.T_meth_rise * (1.0 - np.exp(-t / tox.tau_temp))
        return tox.T_baseline + (T_rise if drug_type == 'METH' else T_rise * 0.3)

    # Sleep state at t=0 (mid-wake)
    P_DAT, _ = sleep_modulation(12.0, arch.sleep, sleep_deprived)

    dt      = 0.5
    n_steps = int(duration / dt)
    t_out   = np.zeros(n_steps)
    De_out  = np.zeros(n_steps)
    Dc_out  = np.zeros(n_steps)
    Dv_out  = np.zeros(n_steps)
    Q_out   = np.zeros(n_steps)
    N_out   = np.zeros(n_steps)
    T_out   = np.zeros(n_steps)
    drug_out = np.zeros(n_steps)

    y = np.array(y0, dtype=float)

    for i in range(n_steps):
        t    = i * dt
        conc = drug_profile(t)
        T    = temp_profile(t)

        dydt = da_terminal_odes(t, y, p, conc, drug_type, T, P_DAT)
        y   += np.array(dydt) * dt
        y    = np.maximum(y, 0.0)
        y[4] = np.clip(y[4], 0.0, 1.0)

        t_out[i]    = t
        De_out[i]   = y[0]
        Dc_out[i]   = y[1]
        Dv_out[i]   = y[2]
        Q_out[i]    = y[3]
        N_out[i]    = y[4]
        T_out[i]    = T
        drug_out[i] = conc

    return {
        't': t_out, 'De': De_out, 'Dc': Dc_out, 'Dv': Dv_out,
        'Q': Q_out, 'N': N_out, 'T': T_out, 'drug': drug_out,
        'De_pct': De_out / p.De_baseline * 100.0,
    }


def simulate_rl_trajectory(arch: NeuralArchitecture,
                            n_trials: int = 500,
                            drug_onset_trial: int = 100,
                            drug_magnitude: float = 3.0) -> Dict:
    """
    Simulate RL learning trajectory across architectures (Layer 4).

    Returns dict with trial-by-trial value estimates and RPE signals.
    """
    rl = arch.rl
    n_states = 3   # neutral, cue, reward
    V  = np.ones(n_states) * rl.w0

    V_history   = np.zeros(n_trials)
    RPE_history = np.zeros(n_trials)
    drug_value  = np.zeros(n_trials)

    # LC gain modulation
    L, NE = arch.lc.I_exc, arch.lc.NE_baseline
    dt    = 1.0

    for trial in range(n_trials):
        L, NE, G = lc_dynamics(L, NE, arch.lc, dt)

        is_drug = trial >= drug_onset_trial
        base_reward = drug_magnitude if is_drug else 1.0
        reward      = base_reward * G  # gain-modulated reward signal

        delta = compute_prediction_error(reward, V[2], V[0], rl.gamma)

        alpha_plus  = rl.alpha0 + rl.k1_alpha * max(0, delta)
        alpha_minus = rl.alpha0 + rl.k2_alpha * max(0, -delta)
        V = update_values(V.copy(), 2, delta, alpha_plus, alpha_minus)

        V_history[trial]   = V[2]
        RPE_history[trial] = delta
        drug_value[trial]  = reward if is_drug else 0.0

    return {
        'trials': np.arange(n_trials),
        'V': V_history,
        'RPE': RPE_history,
        'drug_value': drug_value,
        'drug_onset': drug_onset_trial,
    }


def simulate_chronic_adaptation(arch: NeuralArchitecture,
                                 n_days: int = 365,
                                 dose_per_day: float = 5.0,
                                 use_days: Optional[int] = None) -> Dict:
    """
    Simulate long-timescale neuroadaptation (Layer 6).
    """
    if use_days is None:
        use_days = n_days

    DAT_scale   = 1.0
    setpoint    = arch.terminal.De_baseline
    cp          = arch.chronic

    DAT_history = np.zeros(n_days)
    set_history = np.zeros(n_days)
    De_history  = np.zeros(n_days)

    for day in range(n_days):
        is_using = day < use_days

        if is_using:
            res    = simulate_acute_da_response(arch, 'METH', dose_per_day,
                                                duration=3600)
            De_mean = np.mean(res['De']) * DAT_scale
        else:
            De_mean = arch.terminal.De_baseline * DAT_scale * 0.6

        DAT_scale, setpoint = chronic_adaptation_step(
            DAT_scale, setpoint, De_mean, cp, dt_days=1.0
        )

        DAT_history[day] = DAT_scale
        set_history[day] = setpoint
        De_history[day]  = De_mean

    return {
        'days': np.arange(n_days),
        'DAT_scale': DAT_history,
        'setpoint': set_history,
        'De': De_history,
        'use_days': use_days,
    }


# ─────────────────────────────────────────────────────────
# FIGURE GENERATION
# ─────────────────────────────────────────────────────────

def fig1_acute_da_comparison(output_dir: str):
    """Figure 1: Acute DA response — drug class × architecture comparison."""
    archs = [make_neurotypical(), make_adhd_c(), make_adhd_i()]
    colors_arch = [COLORS['teal'], COLORS['amber'], COLORS['purple']]
    drug_types  = ['none', 'MPH', 'AMPH', 'METH']
    drug_labels = ['No Drug', 'Methylphenidate', 'Amphetamine', 'Methamphetamine']
    drug_concs  = [0.0, 0.5, 2.0, 5.0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for col, (dt, dl, dc) in enumerate(zip(drug_types, drug_labels, drug_concs)):
        for ai, (arch, color) in enumerate(zip(archs, colors_arch)):
            res = simulate_acute_da_response(arch, dt if dc > 0 else 'none',
                                             dc, duration=1800)
            mask = np.arange(0, len(res['t']), max(1, len(res['t']) // 200))
            t_min = res['t'][mask] / 60.0

            axes[0, col].plot(t_min, res['De_pct'][mask],
                              color=color, linewidth=2, label=arch.name, alpha=0.85)
            axes[1, col].plot(t_min, res['N'][mask],
                              color=color, linewidth=2, alpha=0.85)

        axes[0, col].set_title(dl, fontsize=10)
        axes[0, col].set_xlabel('Time (min)')
        axes[1, col].set_xlabel('Time (min)')
        if col == 0:
            axes[0, col].set_ylabel('Extracellular DA (% baseline)')
            axes[1, col].set_ylabel('Terminal Integrity')
        axes[0, col].legend(fontsize=7)
        axes[0, col].axhline(100, color=COLORS['gray'], linestyle='--', alpha=0.4)
        axes[1, col].set_ylim(0, 1.05)

    fig.suptitle('Figure 1: Acute DA Response — Drug Class × Architecture',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_acute_da.png'), bbox_inches='tight')
    plt.close()
    print("  fig1_acute_da.png saved")


def fig2_temperature_bifurcation(output_dir: str):
    """Figure 2: Temperature as bifurcation parameter for neurotoxicity."""
    arch = make_neurotypical()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Quinone accumulation vs temperature
    temps = [37, 38, 39, 39.5, 40, 40.5, 41, 42]
    final_Q = []
    for T in temps:
        arch.toxicity.T_baseline = T
        arch.toxicity.T_meth_rise = 0.0
        res = simulate_acute_da_response(arch, 'METH', 5.0, duration=3600)
        final_Q.append(res['Q'][-1])
    arch.toxicity.T_baseline  = 37.0
    arch.toxicity.T_meth_rise = 3.0

    axes[0].bar(range(len(temps)), final_Q, color=COLORS['red'], alpha=0.8)
    axes[0].set_xticks(range(len(temps)))
    axes[0].set_xticklabels([f'{t}°C' for t in temps], rotation=45, fontsize=9)
    axes[0].set_ylabel('Quinone Accumulation (1h)')
    axes[0].set_title('Temperature → Quinone Formation')

    # Panel B: Terminal integrity over time at different temps
    for T, color in zip([37, 39, 40, 41],
                        [COLORS['teal'], COLORS['amber'], COLORS['orange'], COLORS['red']]):
        arch.toxicity.T_baseline  = T
        arch.toxicity.T_meth_rise = 0.0
        res  = simulate_acute_da_response(arch, 'METH', 5.0, duration=7200)
        mask = np.arange(0, len(res['t']), max(1, len(res['t']) // 300))
        axes[1].plot(res['t'][mask] / 60.0, res['N'][mask],
                     color=color, linewidth=2, label=f'{T}°C')
    arch.toxicity.T_baseline  = 37.0
    arch.toxicity.T_meth_rise = 3.0

    axes[1].set_xlabel('Time (min)')
    axes[1].set_ylabel('Terminal Integrity')
    axes[1].set_title('Temperature-Gated Terminal Damage')
    axes[1].legend()

    # Panel C: Phase diagram (temperature × cytoplasmic DA → oxidation rate)
    T_range  = np.linspace(37, 43, 100)
    Dc_range = np.linspace(0, 50, 100)
    T_grid, Dc_grid = np.meshgrid(T_range, Dc_range)
    ox_rate = 0.001 * 2.5**((T_grid - 37.0) / 10.0) * Dc_grid

    im = axes[2].contourf(T_grid, Dc_grid, np.log10(ox_rate + 1e-10),
                          levels=20, cmap='inferno')
    axes[2].axvline(x=40, color=COLORS['white'], linestyle='--', alpha=0.5,
                    label='~40°C threshold')
    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_ylabel('Cytoplasmic DA (µM)')
    axes[2].set_title('Oxidation Rate (log₁₀)')
    axes[2].legend()
    plt.colorbar(im, ax=axes[2])

    fig.suptitle('Figure 2: Temperature as Bifurcation Parameter for Neurotoxicity',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_temperature.png'), bbox_inches='tight')
    plt.close()
    print("  fig2_temperature.png saved")


def fig3_rl_trajectories(output_dir: str):
    """Figure 3: RL learning trajectories across architectures."""
    archs  = [make_neurotypical(), make_adhd_c(), make_adhd_i()]
    colors = [COLORS['teal'], COLORS['amber'], COLORS['purple']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for arch, color in zip(archs, colors):
        res = simulate_rl_trajectory(arch, n_trials=500,
                                      drug_onset_trial=100, drug_magnitude=3.0)
        t = res['trials']
        axes[0].plot(t, res['V'],   color=color, linewidth=1.5, label=arch.name, alpha=0.85)
        axes[1].plot(t, res['RPE'], color=color, linewidth=1.0, alpha=0.6)
        # Smoothed RPE
        kernel = np.ones(20) / 20
        axes[2].plot(t, np.convolve(np.abs(res['RPE']), kernel, 'same'),
                     color=color, linewidth=2, label=arch.name)

    for ax in axes:
        ax.axvline(100, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Drug onset')
    axes[0].set_title('Value Estimates V(s)')
    axes[0].set_ylabel('V')
    axes[1].set_title('Prediction Error (RPE)')
    axes[1].set_ylabel('δ')
    axes[2].set_title('|RPE| — Smoothed (20-trial window)')
    axes[2].set_ylabel('|δ|')
    for ax in axes:
        ax.set_xlabel('Trial')
        ax.legend(fontsize=8)

    fig.suptitle('Figure 3: RL Learning Trajectories — TD-Learning Layer',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_rl_trajectories.png'), bbox_inches='tight')
    plt.close()
    print("  fig3_rl_trajectories.png saved")


def fig4_chronic_adaptation(output_dir: str):
    """Figure 4: Chronic neuroadaptation — DAT upregulation and setpoint drift."""
    archs  = [make_neurotypical(), make_adhd_c(), make_adhd_i()]
    colors = [COLORS['teal'], COLORS['amber'], COLORS['purple']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for arch, color in zip(archs, colors):
        res = simulate_chronic_adaptation(arch, n_days=365,
                                          dose_per_day=5.0, use_days=180)
        days = res['days']
        axes[0].plot(days, res['DAT_scale'], color=color, linewidth=2, label=arch.name)
        axes[1].plot(days, res['setpoint'] / arch.terminal.De_baseline * 100,
                     color=color, linewidth=2)
        axes[2].plot(days, res['De'],        color=color, linewidth=2)

    for ax in axes:
        ax.axvline(180, color=COLORS['gray'], linestyle='--', alpha=0.5, label='Cessation')
        ax.set_xlabel('Days')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('DAT Scale Factor')
    axes[0].set_title('DAT Upregulation')
    axes[1].set_ylabel('Setpoint (% baseline)')
    axes[1].set_title('Allostatic Setpoint Drift')
    axes[2].set_ylabel('Mean [DA]e (µM)')
    axes[2].set_title('Extracellular DA Level')

    fig.suptitle('Figure 4: Chronic Neuroadaptation — 365-Day Simulation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_chronic.png'), bbox_inches='tight')
    plt.close()
    print("  fig4_chronic.png saved")


def fig5_sensitivity_analysis(output_dir: str):
    """Figure 5: One-at-a-time sensitivity analysis."""
    arch_base = make_neurotypical()

    params_to_vary = {
        'Vmax': ('terminal', 'Vmax', np.linspace(2.0, 6.0, 20)),
        'k_rev': ('terminal', 'k_rev', np.linspace(2.0, 15.0, 20)),
        'Q10': ('toxicity', 'Q10', np.linspace(1.5, 4.0, 20)),
        'LC I_exc': ('lc', 'I_exc', np.linspace(1.0, 4.5, 20)),
        'alpha0': ('rl', 'alpha0', np.linspace(0.02, 0.20, 20)),
        'k_deg': ('terminal', 'k_deg', np.linspace(0.00005, 0.0005, 20)),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (pname, (layer, attr, vals)) in zip(axes, params_to_vary.items()):
        outcomes = []
        for v in vals:
            import copy
            arch = copy.deepcopy(arch_base)
            setattr(getattr(arch, layer), attr, v)
            res = simulate_acute_da_response(arch, 'METH', 5.0, duration=3600)
            outcomes.append(np.max(res['De_pct']))

        ax.plot(vals, outcomes, color=COLORS['amber'], linewidth=2)
        ax.set_xlabel(pname)
        ax.set_ylabel('Peak [DA]e (% baseline)')
        ax.set_title(f'Sensitivity: {pname}')
        ax.axhline(np.mean(outcomes), color=COLORS['gray'], linestyle='--', alpha=0.5)

    fig.suptitle('Figure 5: One-at-a-Time Sensitivity Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_sensitivity.png'), bbox_inches='tight')
    plt.close()
    print("  fig5_sensitivity.png saved")


def fig6_sleep_drug_interaction(output_dir: str):
    """Figure 6: Sleep deprivation × drug interaction."""
    arch = make_neurotypical()
    conditions = [
        ('Rested + No Drug',     False, 'none', 0.0,  COLORS['teal']),
        ('Rested + METH',        False, 'METH', 5.0,  COLORS['amber']),
        ('Sleep-Deprived + METH', True, 'METH', 5.0,  COLORS['red']),
        ('Sleep-Deprived + MPH', True,  'MPH',  0.5,  COLORS['purple']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for label, sd, dt, dc, color in conditions:
        res  = simulate_acute_da_response(arch, dt if dc > 0 else 'none',
                                          dc, duration=3600, sleep_deprived=sd)
        mask = np.arange(0, len(res['t']), max(1, len(res['t']) // 250))
        t_min = res['t'][mask] / 60.0

        axes[0].plot(t_min, res['De_pct'][mask], color=color, linewidth=2,
                     label=label, alpha=0.85)
        axes[1].plot(t_min, res['N'][mask],      color=color, linewidth=2, alpha=0.85)
        axes[2].plot(t_min, res['Q'][mask],       color=color, linewidth=2, alpha=0.85)

    axes[0].set_ylabel('[DA]e (% baseline)')
    axes[1].set_ylabel('Terminal Integrity')
    axes[2].set_ylabel('Quinone Accumulation')

    for ax in axes:
        ax.set_xlabel('Time (min)')
        ax.legend(fontsize=7)

    axes[0].set_title('Extracellular DA')
    axes[1].set_title('Terminal Integrity')
    axes[2].set_title('Oxidative Stress')

    fig.suptitle('Figure 6: Sleep Deprivation × Drug Interaction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_sleep_drug.png'), bbox_inches='tight')
    plt.close()
    print("  fig6_sleep_drug.png saved")


def fig7_architecture_comparison(output_dir: str):
    """Figure 7: Cross-architecture summary comparison."""
    archs       = [make_neurotypical(), make_adhd_c(), make_adhd_i()]
    colors_arch = [COLORS['teal'], COLORS['amber'], COLORS['purple']]
    names       = [a.name for a in archs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Peak DA across drug conditions
    drug_types = ['none', 'MPH', 'AMPH', 'METH']
    drug_concs = [0.0,    0.5,   2.0,    5.0  ]
    x = np.arange(len(drug_types))
    width = 0.25

    for ai, (arch, color) in enumerate(zip(archs, colors_arch)):
        peaks = []
        for dt, dc in zip(drug_types, drug_concs):
            res = simulate_acute_da_response(arch, dt if dc > 0 else 'none',
                                             dc, duration=1800)
            peaks.append(np.max(res['De_pct']))
        axes[0, 0].bar(x + ai * width, peaks, width, color=color,
                       label=arch.name, alpha=0.85)

    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(drug_types)
    axes[0, 0].set_ylabel('Peak [DA]e (% baseline)')
    axes[0, 0].set_title('Peak DA Response by Architecture')
    axes[0, 0].legend()

    # Panel B: Final terminal integrity after METH
    integrities = []
    for arch in archs:
        res = simulate_acute_da_response(arch, 'METH', 5.0, duration=7200)
        integrities.append(res['N'][-1])

    bars = axes[0, 1].bar(names, integrities, color=colors_arch, alpha=0.85)
    axes[0, 1].set_ylabel('Terminal Integrity (2h METH)')
    axes[0, 1].set_title('Residual Terminal Integrity')
    axes[0, 1].set_ylim(0, 1.1)
    for bar, val in zip(bars, integrities):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                        f'{val:.3f}', ha='center', fontsize=9)

    # Panel C: RL value escalation
    for arch, color in zip(archs, colors_arch):
        res = simulate_rl_trajectory(arch)
        kernel = np.ones(15) / 15
        smoothed = np.convolve(res['V'], kernel, 'same')
        axes[1, 0].plot(res['trials'], smoothed, color=color,
                        linewidth=2, label=arch.name)
    axes[1, 0].axvline(100, color=COLORS['gray'], linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Trial')
    axes[1, 0].set_ylabel('V(s) — Smoothed')
    axes[1, 0].set_title('Value Escalation After Drug Onset')
    axes[1, 0].legend()

    # Panel D: Normalized architecture parameter profiles
    categories = ['DAT Vmax', 'LC I_exc', 'α2 Gain', 'RL α', 'RL β', 'RL w₀']
    for ai, (arch, color) in enumerate(zip(archs, colors_arch)):
        values = [
            arch.terminal.Vmax / 4.0,
            arch.lc.I_exc / 4.0,
            arch.lc.g_alpha2,
            arch.rl.alpha0 / 0.15,
            arch.rl.beta / 6.0,
            arch.rl.w0,
        ]
        axes[1, 1].plot(range(len(values)), values, 'o-',
                        color=color, linewidth=2, markersize=8,
                        label=arch.name, alpha=0.85)

    axes[1, 1].set_xticks(range(len(categories)))
    axes[1, 1].set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Normalized Parameter Value')
    axes[1, 1].set_title('Architecture Parameter Profiles')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 1.2)

    fig.suptitle('Figure 7: Cross-Architecture Comparison Summary',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig7_comparison.png'), bbox_inches='tight')
    plt.close()
    print("  fig7_comparison.png saved")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def generate_all_figures(output_dir: str = './figures'):
    """Generate all seven main-text figures."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Figure 1: Acute DA comparison...")
    fig1_acute_da_comparison(output_dir)
    print("Generating Figure 2: Temperature bifurcation...")
    fig2_temperature_bifurcation(output_dir)
    print("Generating Figure 3: RL trajectories...")
    fig3_rl_trajectories(output_dir)
    print("Generating Figure 4: Chronic adaptation...")
    fig4_chronic_adaptation(output_dir)
    print("Generating Figure 5: Sensitivity analysis...")
    fig5_sensitivity_analysis(output_dir)
    print("Generating Figure 6: Sleep-drug interaction...")
    fig6_sleep_drug_interaction(output_dir)
    print("Generating Figure 7: Architecture comparison...")
    fig7_architecture_comparison(output_dir)
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    generate_all_figures()
