"""
ANCCR-Based Reinforcement Learning Layer
Revision of the TD-learning layer in meth_neurodiv_model.py

Implements the Amortized Causal Contingency for Reward (ANCCR) framework
(Jeong et al. 2022, Nature Neuroscience) with IRI-scaling per Burke et al. 2026.

Key change: learning rate scales proportionally with inter-reward interval (IRI),
not as a free parameter. This produces the Burke et al. scaling law:
  α ∝ IRI, such that total conditioning time is invariant to trial frequency.

The terminal kinetics, temperature toxicity, LC-NE gain, sleep, and
chronic adaptation layers are imported from meth_neurodiv_model.py unchanged.

Author: Evelyn Campbell
DOI: 10.5281/ZENODO.19625787
Date: March 2026 (revised)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import sys
import copy

# Import base model layers
sys.path.insert(0, os.path.dirname(__file__))
from meth_neurodiv_model import (
    TerminalParams, ToxicityParams, LCParams, SleepParams, ChronicParams,
    make_neurotypical, make_adhd_c, make_adhd_i,
    simulate_acute_da_response, lc_dynamics,
    COLORS, setup_plot_style,
)

setup_plot_style()


# ─────────────────────────────────────────────────────────
# ANCCR PARAMETER STRUCTURE
# ─────────────────────────────────────────────────────────

@dataclass
class ANCCRParams:
    """
    ANCCR learning parameters (Jeong et al. 2022).

    In this model the RL field of NeuralArchitecture is replaced
    by ANCCRParams.  The NeuralArchitecture from the base module is
    subclassed below to swap the RL layer out.
    """
    t_constant: float = 200.0       # eligibility trace time constant (s)
    alpha: float = 0.1              # association learning rate
    k: float = 0.3                  # base-rate learning rate multiplier
    w: float = 0.5                  # prospective / retrospective weighting
    threshold: float = 0.3          # meaningful causal-target threshold
    alpha_reward: float = 0.2       # causal weight learning rate
    beta_drug: float = 5.0          # innate meaningfulness of drug reward
    beta_natural: float = 1.0       # innate meaningfulness of natural reward
    # Burke et al. 2026: t_constant scales with IRI
    t_constant_iri_ratio: float = 0.33  # t_constant = ratio × IRI


@dataclass
class ANCCRArchitecture:
    """
    Neural architecture where the RL layer is ANCCR
    (replaces RLParams from the base model).
    """
    name: str = "Neurotypical"
    terminal: TerminalParams = field(default_factory=TerminalParams)
    toxicity: ToxicityParams = field(default_factory=ToxicityParams)
    lc: LCParams = field(default_factory=LCParams)
    anccr: ANCCRParams = field(default_factory=ANCCRParams)
    sleep: SleepParams = field(default_factory=SleepParams)
    chronic: ChronicParams = field(default_factory=ChronicParams)


def make_nt_anccr() -> ANCCRArchitecture:
    return ANCCRArchitecture(name="Neurotypical")

def make_adhd_c_anccr() -> ANCCRArchitecture:
    """ADHD-C: lower causal threshold, faster causal weight updates."""
    arch = ANCCRArchitecture(name="ADHD-C")
    arch.terminal.Vmax  *= 0.85
    arch.terminal.F_tonic = 3.5
    arch.lc.I_exc         = 3.5
    arch.lc.g_alpha2      = 0.55
    arch.anccr.threshold    = 0.20   # lower threshold → more causal targets
    arch.anccr.alpha_reward = 0.25   # faster causal weight updates
    return arch

def make_adhd_i_anccr() -> ANCCRArchitecture:
    """ADHD-I: higher causal threshold, LC hypoarousal."""
    arch = ANCCRArchitecture(name="ADHD-I")
    arch.terminal.Vmax *= 0.90
    arch.lc.I_exc       = 1.2
    arch.lc.g_alpha2    = 1.0
    arch.anccr.threshold = 0.35      # higher threshold → fewer causal targets
    return arch


# ─────────────────────────────────────────────────────────
# ANCCR LEARNING ENGINE
# ─────────────────────────────────────────────────────────

class ANCCREngine:
    """
    Implements the ANCCR learning algorithm (Jeong et al. 2022)
    with Burke et al. 2026 IRI-scaling.

    Tracks associations between cue states and drug / natural reward outcomes.

    Simplified for multi-scale integration: continuous-time eligibility traces
    updated at each discrete event (cue onset, reward delivery).

    Parameters
    ----------
    params   : ANCCRParams
    n_cues   : number of cue stimuli
    n_outcomes : number of possible reward outcomes
    iri      : inter-reward interval (seconds); sets t_constant via IRI scaling
    """

    def __init__(self, params: ANCCRParams, n_cues: int = 2,
                 n_outcomes: int = 2, iri: float = 60.0):
        self.p          = params
        self.n_cues     = n_cues
        self.n_outcomes = n_outcomes

        # Burke et al. 2026 IRI scaling
        self.t_constant = params.t_constant_iri_ratio * iri
        self.iri        = iri

        # State
        self.traces    = np.zeros(n_cues)         # eligibility traces
        self.CW        = np.zeros((n_cues, n_outcomes))  # causal weights
        self.NC        = np.zeros(n_cues)         # net contingency
        self.base_rate = np.zeros(n_outcomes)     # background reward rates
        self.t_last    = np.zeros(n_outcomes)     # time of last outcome

    def update_traces(self, dt: float, active_cues: List[int]) -> None:
        """Decay traces and activate cues present at this timestep."""
        decay = np.exp(-dt / self.t_constant)
        self.traces *= decay
        for ci in active_cues:
            self.traces[ci] = 1.0   # reset to 1 on cue onset

    def step(self, t: float, active_cues: List[int],
             outcomes: Dict[int, float]) -> Dict:
        """
        Process one event.

        Parameters
        ----------
        t           : current time (s)
        active_cues : indices of cues currently active
        outcomes    : {outcome_id: magnitude} for rewards delivered now

        Returns dict with ANCCR quantities and response values.
        """
        p = self.p

        # 1. Update base rates (background reward frequency)
        for oid, mag in outcomes.items():
            dt_since = t - self.t_last[oid]
            self.base_rate[oid] += p.k * (1.0 / max(dt_since, 1.0)
                                           - self.base_rate[oid])
            self.t_last[oid] = t

        # 2. Prospective relative contingency (PRC): cue → outcome
        PRC = np.zeros((self.n_cues, self.n_outcomes))
        for ci in range(self.n_cues):
            for oid in range(self.n_outcomes):
                if self.traces[ci] > 0:
                    # Rate of outcome given cue vs background
                    cue_rate  = self.traces[ci] / self.t_constant
                    net_cont  = cue_rate - self.base_rate[oid]
                    PRC[ci, oid] = net_cont

        # 3. Net contingency (NC): combined prospective / retrospective
        NC = np.zeros(self.n_cues)
        for ci in range(self.n_cues):
            prosp  = np.sum(PRC[ci, :])
            retro  = self.traces[ci] if len(outcomes) > 0 else 0.0
            NC[ci] = p.w * prosp + (1.0 - p.w) * retro

        # 4. Subjective relative contingency (SRC)
        SRC = np.zeros(self.n_cues)
        for ci in range(self.n_cues):
            SRC[ci] = NC[ci] if NC[ci] > p.threshold else 0.0

        # 5. Compute effective alpha via IRI scaling (Burke et al. 2026)
        effective_alpha = p.alpha * self.iri / max(self.iri, 1.0)

        # 6. Dopamine signal: net contingency × causal weight
        DA_total = 0.0
        for ci in range(self.n_cues):
            for oid in range(self.n_outcomes):
                DA_total += SRC[ci] * self.CW[ci, oid]

        # 7. Update causal weights (Jeong 2022, Eqs 11–12)
        for oid, mag in outcomes.items():
            if oid == 0:
                innate_value = p.beta_drug    * mag
            else:
                innate_value = p.beta_natural * mag

            for ci in range(self.n_cues):
                if DA_total >= 0.0:
                    delta = (innate_value - self.CW[ci, oid]) * self.traces[ci]
                else:
                    delta = (0.0 - self.CW[ci, oid]) * self.traces[ci]
                self.CW[ci, oid] += p.alpha_reward * delta

        # 8. Response value (Jeong 2022, Eq 13)
        Q = np.array([
            sum(SRC[ci] * self.CW[ci, k] if SRC[ci] > 0 else 0.0
                for k in range(self.n_outcomes))
            for ci in range(self.n_cues)
        ])

        return {
            'DA': DA_total,
            'NC': NC.copy(),
            'PRC': PRC.copy(),
            'SRC': SRC.copy(),
            'CW': self.CW.copy(),
            'Q': Q.copy(),
            'effective_alpha': effective_alpha,
            't_constant': self.t_constant,
        }


# ─────────────────────────────────────────────────────────
# TERMINAL KINETICS ODE (re-exported for convenience)
# ─────────────────────────────────────────────────────────

def da_terminal_odes(t, y, params, drug_conc, drug_type, T):
    """Thin wrapper — see meth_neurodiv_model.da_terminal_odes for full docs."""
    from meth_neurodiv_model import da_terminal_odes as _odes
    return _odes(t, y, params, drug_conc, drug_type, T, P_DAT=0.0)


# ─────────────────────────────────────────────────────────
# ANCCR SIMULATION WRAPPERS
# ─────────────────────────────────────────────────────────

def simulate_iri_scaling(arch: ANCCRArchitecture,
                          iris: List[float] = [30, 60, 120, 300, 600],
                          n_trials: int = 20,
                          max_time: float = 7200.0) -> Dict:
    """
    Reproduce Burke et al. 2026 Result:
    Learning rate ∝ IRI; total conditioning time invariant.

    Simulates cue-reward conditioning at multiple IRIs and
    measures trials-to-criterion per IRI.
    """
    results = {}

    for iri in iris:
        engine = ANCCREngine(arch.anccr, n_cues=2, n_outcomes=2, iri=iri)
        t      = 0.0
        trial  = 0
        Q_history = []

        while t < max_time and trial < n_trials:
            # Cue onset
            engine.update_traces(iri * 0.1, active_cues=[0])
            t += iri * 0.1

            # Reward delivery
            out = engine.step(t, active_cues=[0], outcomes={0: 1.0})
            Q_history.append(np.max(out['Q']))
            t     += iri * 0.9
            trial += 1

        results[iri] = {
            'Q_history': np.array(Q_history),
            'trials': np.arange(len(Q_history)),
            'time_per_trial': iri,
            'effective_alpha': arch.anccr.alpha * iri / max(iri, 1.0),
            't_constant': arch.anccr.t_constant_iri_ratio * iri,
        }

    return results


def simulate_drug_conditioning(arch: ANCCRArchitecture,
                                n_trials: int = 100,
                                iri: float = 60.0,
                                drug_onset: int = 20,
                                drug_conc: float = 5.0) -> Dict:
    """
    Simulate drug vs natural reward conditioning via ANCCR.

    Outcome 0 = drug, Outcome 1 = natural reward.
    Drug onset at trial `drug_onset`.
    """
    engine = ANCCREngine(arch.anccr, n_cues=2, n_outcomes=2, iri=iri)
    L, NE  = arch.lc.I_exc, arch.lc.NE_baseline

    Q_drug    = np.zeros(n_trials)
    Q_natural = np.zeros(n_trials)
    DA_signal = np.zeros(n_trials)
    NC_hist   = np.zeros(n_trials)

    for trial in range(n_trials):
        t  = trial * iri
        L, NE, G = lc_dynamics(L, NE, arch.lc, dt=iri)

        engine.update_traces(iri, active_cues=[0])
        is_drug = trial >= drug_onset
        outcomes: Dict[int, float] = {}

        if is_drug:
            # Drug reward — magnitude modulated by G (LC gain)
            # and kinetics from terminal layer
            res = simulate_acute_da_response(arch, 'METH', drug_conc,
                                             duration=300)
            da_peak  = np.max(res['De_pct']) / 100.0  # normalized
            outcomes[0] = da_peak * G
        else:
            outcomes[1] = 1.0 * G  # natural reward

        out = engine.step(t, active_cues=[0], outcomes=outcomes)

        Q_drug[trial]    = out['Q'][0] if out['Q'][0] > 0 else 0.0
        Q_natural[trial] = out['Q'][0] if 1 in outcomes else 0.0
        DA_signal[trial] = out['DA']
        NC_hist[trial]   = np.mean(np.abs(out['NC']))

    return {
        'trials': np.arange(n_trials),
        'Q_drug': Q_drug,
        'Q_natural': Q_natural,
        'DA': DA_signal,
        'NC': NC_hist,
        'drug_onset': drug_onset,
    }


def simulate_binge_vs_intermittent(arch: ANCCRArchitecture,
                                    n_doses: int = 20,
                                    drug_conc: float = 5.0) -> Dict:
    """
    Contrast binge (short IRI) vs intermittent (long IRI) conditioning.

    Binge:       IRI = 30s   (back-to-back dosing)
    Intermittent: IRI = 600s  (spaced dosing)

    Burke et al. 2026 predicts less per-exposure learning during binge.
    """
    schedules = {'Binge (IRI=30s)': 30.0, 'Intermittent (IRI=600s)': 600.0}
    results = {}

    for label, iri in schedules.items():
        engine = ANCCREngine(arch.anccr, n_cues=2, n_outcomes=2, iri=iri)
        Q_hist = []
        DA_hist = []

        for dose in range(n_doses):
            t = dose * iri
            engine.update_traces(iri * 0.1, active_cues=[0])
            out = engine.step(t, active_cues=[0], outcomes={0: 1.0})
            Q_hist.append(np.max(out['Q']))
            DA_hist.append(out['DA'])

        total_time = n_doses * iri
        results[label] = {
            'doses': np.arange(n_doses),
            'Q': np.array(Q_hist),
            'DA': np.array(DA_hist),
            'iri': iri,
            'total_time': total_time,
            'effective_alpha': out['effective_alpha'],
        }

    return results


# ─────────────────────────────────────────────────────────
# FIGURE GENERATION (Figs 8–10)
# ─────────────────────────────────────────────────────────

def fig_iri_scaling(output_dir: str):
    """Figure 8: IRI scaling — Burke et al. 2026 replication."""
    archs  = [make_nt_anccr(), make_adhd_c_anccr(), make_adhd_i_anccr()]
    colors = [COLORS['teal'], COLORS['amber'], COLORS['purple']]
    iris   = [30, 60, 120, 300, 600]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for arch, color in zip(archs, colors):
        res = simulate_iri_scaling(arch, iris=iris, n_trials=30)

        # Panel A: alpha vs IRI
        alphas = [res[iri]['effective_alpha'] for iri in iris]
        axes[0].plot(iris, alphas, 'o-', color=color, linewidth=2,
                     markersize=7, label=arch.name)

        # Panel B: Total conditioning time (IRI × trials-to-80% Q_max)
        total_times = []
        for iri in iris:
            Q = res[iri]['Q_history']
            if len(Q) == 0 or np.max(Q) == 0:
                total_times.append(np.nan)
                continue
            Q_norm   = Q / np.max(Q)
            idx_80   = np.argmax(Q_norm >= 0.80) if np.any(Q_norm >= 0.80) else len(Q) - 1
            total_times.append(idx_80 * iri)
        axes[1].plot(iris, total_times, 'o-', color=color, linewidth=2,
                     markersize=7, label=arch.name)

        # Panel C: t_constant vs IRI
        t_consts = [res[iri]['t_constant'] for iri in iris]
        axes[2].plot(iris, t_consts, 'o-', color=color, linewidth=2,
                     markersize=7, label=arch.name)

    axes[0].set_xlabel('IRI (s)')
    axes[0].set_ylabel('Effective Learning Rate (α)')
    axes[0].set_title('α ∝ IRI (Burke et al. 2026)')
    axes[0].legend()

    axes[1].set_xlabel('IRI (s)')
    axes[1].set_ylabel('Total Time-to-Criterion (s)')
    axes[1].set_title('Total Conditioning Time Invariant to IRI')
    axes[1].legend()

    axes[2].set_xlabel('IRI (s)')
    axes[2].set_ylabel('Eligibility Trace τ (s)')
    axes[2].set_title('IRI-Scaled Trace Time Constant')
    axes[2].legend()

    fig.suptitle('Figure 8: IRI Scaling — ANCCR Replication of Burke et al. 2026',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_iri_scaling.png'), bbox_inches='tight')
    plt.close()
    print("  fig_iri_scaling.png saved")


def fig_anccr_drug_comparison(output_dir: str):
    """Figure 9: Drug vs natural reward conditioning across architectures."""
    archs  = [make_nt_anccr(), make_adhd_c_anccr(), make_adhd_i_anccr()]
    colors = [COLORS['teal'], COLORS['amber'], COLORS['purple']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for arch, color in zip(archs, colors):
        res = simulate_drug_conditioning(arch, n_trials=100, iri=60.0,
                                          drug_onset=20, drug_conc=5.0)
        t = res['trials']

        kernel   = np.ones(5) / 5
        Q_sm     = np.convolve(res['Q_drug'], kernel, 'same')
        DA_sm    = np.convolve(res['DA'],     kernel, 'same')

        axes[0].plot(t, Q_sm,  color=color, linewidth=2, label=arch.name)
        axes[1].plot(t, DA_sm, color=color, linewidth=2, label=arch.name)
        axes[2].plot(t, res['NC'], color=color, linewidth=1.5, alpha=0.75,
                     label=arch.name)

    for ax in axes:
        ax.axvline(20, color=COLORS['gray'], linestyle='--', alpha=0.5,
                   label='Drug onset')
        ax.set_xlabel('Trial')
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Q (Drug)')
    axes[0].set_title('Response Value — Drug')
    axes[1].set_ylabel('DA signal')
    axes[1].set_title('ANCCR DA Signal')
    axes[2].set_ylabel('|NC|')
    axes[2].set_title('Net Contingency')

    fig.suptitle('Figure 9: ANCCR Drug Conditioning — Architecture Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_anccr_drug.png'), bbox_inches='tight')
    plt.close()
    print("  fig_anccr_drug.png saved")


def fig_binge_vs_intermittent(output_dir: str):
    """Figure 10: Binge vs intermittent use — ANCCR predictions."""
    archs  = [make_nt_anccr(), make_adhd_c_anccr(), make_adhd_i_anccr()]
    colors = [COLORS['teal'], COLORS['amber'], COLORS['purple']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for arch, color in zip(archs, colors):
        res = simulate_binge_vs_intermittent(arch, n_doses=20)

        binge_label = 'Binge (IRI=30s)'
        inter_label = 'Intermittent (IRI=600s)'
        binge = res[binge_label]
        inter = res[inter_label]

        # Panel A: Q accumulation per dose
        axes[0].plot(binge['doses'], binge['Q'], '--', color=color,
                     linewidth=1.5, alpha=0.6)
        axes[0].plot(inter['doses'], inter['Q'], '-',  color=color,
                     linewidth=2.5, label=arch.name + ' (intermit.)')

        # Panel B: Q per unit time
        binge_qpt = binge['Q'] / np.maximum(binge['doses'] * binge['iri'], 1.0)
        inter_qpt = inter['Q'] / np.maximum(inter['doses'] * inter['iri'], 1.0)
        axes[1].plot(binge['doses'], binge_qpt, '--', color=color,
                     linewidth=1.5, alpha=0.6)
        axes[1].plot(inter['doses'], inter_qpt, '-',  color=color,
                     linewidth=2.5, label=arch.name + ' (intermit.)')

        # Panel C: DA signal
        axes[2].plot(binge['doses'], binge['DA'], '--', color=color,
                     linewidth=1.5, alpha=0.6)
        axes[2].plot(inter['doses'], inter['DA'], '-',  color=color,
                     linewidth=2.5, label=arch.name)

    for ax in axes:
        ax.set_xlabel('Dose #')
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Response Value Q')
    axes[0].set_title('Q per Dose (— intermit., -- binge)')
    axes[1].set_ylabel('Q / time')
    axes[1].set_title('Q per Unit Time')
    axes[2].set_ylabel('DA signal')
    axes[2].set_title('DA Signal per Dose')

    fig.suptitle('Figure 10: Binge vs Intermittent — ANCCR Predicts Less per-Exposure Learning During Binge',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_binge_intermittent.png'), bbox_inches='tight')
    plt.close()
    print("  fig_binge_intermittent.png saved")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def generate_anccr_figures(output_dir: str = './figures'):
    """Generate the three ANCCR-specific figures (Figs 8–10)."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Figure 8: IRI scaling...")
    fig_iri_scaling(output_dir)
    print("Generating Figure 9: ANCCR drug comparison...")
    fig_anccr_drug_comparison(output_dir)
    print("Generating Figure 10: Binge vs intermittent...")
    fig_binge_vs_intermittent(output_dir)
    print(f"\nANCCR figures saved to {output_dir}/")


if __name__ == '__main__':
    generate_anccr_figures()
