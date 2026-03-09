import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons
from scipy.integrate import odeint
import threading

# ============================================================
# 1. PARAMÈTRES PHYSIQUES
# ============================================================
omega_x = 8.0    # assez grand pour que ω_c = ω_x + δ > 0 sur tout [-7, 7]
wc0     = omega_x

G0_INIT = 1.5   # splitting 2g visible à δ=0, analogue au fort couplage de l'article
G0_MIN  = 0.01
G0_MAX  = 3.5

def g_of_wc(wc, g0):
    """g ∝ sqrt(wc/wc0), normalisé pour rester comparable à δ"""
    return g0 * np.sqrt(wc / wc0)

# ============================================================
# 2. PLAGES — en détuning δ = ω_c - ω_x, centré sur 0
# ============================================================
DELTA_MAX      = 7.0
DELTA_MIN      = -DELTA_MAX
DELTA_MIN_SAFE = max(DELTA_MIN, -omega_x + 0.1)

delta_range   = np.linspace(DELTA_MIN_SAFE, DELTA_MAX, 300)
delta_targets = np.linspace(DELTA_MIN_SAFE, DELTA_MAX, 100)
frames        = 300

omega_c_range   = omega_x + delta_range
omega_c_targets = omega_x + delta_targets

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================
WC_MAX           = omega_x + DELTA_MAX
AMP_MIN, AMP_MAX = 0.03, 0.13

def normalize_amp(value, val_max):
    return AMP_MIN + (AMP_MAX - AMP_MIN) * np.clip(value / val_max, 0, 1)

def get_spring(start, end, amplitude=0.08):
    xs = np.linspace(start, end, 100)
    ys = amplitude * np.sin(np.linspace(0, 20 * np.pi, 100))
    return xs, ys

def system_dynamics(y, t, w_c, w_x, g):
    x1, x2, v1, v2 = y
    f1 = -(w_c**2) * x1 + g * (x2 - x1)
    f2 = -(w_x**2) * x2 - g * (x2 - x1)
    return [v1, v2, f1, f2]

# ============================================================
# 4. CALCUL SPECTRE ET SIMULATIONS
# ============================================================
def compute_spectrum(g0):
    plus_curve, moins_curve = [], []
    for w_c in omega_c_range:
        g = g_of_wc(w_c, g0)
        H = np.array([[w_c, g], [g, omega_x]])
        freqs = np.sort(np.linalg.eigvals(H))
        moins_curve.append(freqs[0])
        plus_curve.append(freqs[1])
    return np.array(plus_curve), np.array(moins_curve)

def compute_simulations(g0):
    n_frames = max(60, int(300 * G0_INIT / g0))
    T_rabi   = 2 * np.pi / g0 if g0 > 0.01 else 20.0
    t_seq    = np.linspace(0, 3 * T_rabi, n_frames)
    amp0     = min(0.6, 0.6 * G0_INIT / g0)

    s_photon, s_minus, s_plus, s_unc = [], [], [], []
    for wc in omega_c_targets:
        g = g_of_wc(wc, g0)
        H = np.array([[wc, g], [g, omega_x]])
        evals, evecs = np.linalg.eigh(H)

        v_minus = evecs[:, 0] * (amp0 / np.max(np.abs(evecs[:, 0])))
        v_plus  = evecs[:, 1] * (amp0 / np.max(np.abs(evecs[:, 1])))

        s_photon.append(odeint(system_dynamics, [amp0, 0.0, 0.0, 0.0],          t_seq, args=(wc, omega_x, g)))
        s_minus.append( odeint(system_dynamics, [v_minus[0], v_minus[1], 0, 0],  t_seq, args=(wc, omega_x, g)))
        s_plus.append(  odeint(system_dynamics, [v_plus[0],  v_plus[1],  0, 0],  t_seq, args=(wc, omega_x, g)))
        # Fantômes: chaque masse oscille librement à sa propre fréquence (g=0)
        # masse 1 (photon) excitée, masse 2 (exciton) excitée indépendamment
        sol_unc_1 = odeint(system_dynamics, [amp0, 0.0, 0.0, 0.0], t_seq, args=(wc, omega_x, 0))
        sol_unc_2 = odeint(system_dynamics, [0.0, amp0, 0.0, 0.0], t_seq, args=(wc, omega_x, 0))
        # On combine: col0 = photon libre, col1 = exciton libre
        sol_unc_combined = np.column_stack([sol_unc_1[:, 0], sol_unc_2[:, 1],
                                            sol_unc_1[:, 2], sol_unc_2[:, 3]])
        s_unc.append(sol_unc_combined)

    return s_photon, s_minus, s_plus, s_unc, n_frames

# Calcul initial
plus_curve, moins_curve = compute_spectrum(G0_INIT)
sols_photon, sols_minus, sols_plus, sols_unc, n_frames_init = compute_simulations(G0_INIT)

# ============================================================
# 5. FIGURE & LAYOUT
# ============================================================
fig, (ax_phys, ax_spec) = plt.subplots(
    2, 1, figsize=(11, 9.5),
    gridspec_kw={'height_ratios': [1, 1.4]}
)
fig.patch.set_facecolor('#ffffff')
plt.subplots_adjust(bottom=0.30, hspace=0.35)

# --- Panneau physique ---
L           = 3
eq_cav      = L
eq_qd       = 2 * L
mur_gauche  = -0.2
mur_droit   = 2.5 * L

ax_phys.set_xlim(mur_gauche - 0.5, mur_droit + 0.5)
ax_phys.set_ylim(-1, 1.3)
ax_phys.set_aspect('equal')
ax_phys.axis('off')

ghost1 = ax_phys.add_patch(plt.Circle((eq_cav, 0), 0.22, color='blue', alpha=0.15))
ghost2 = ax_phys.add_patch(plt.Circle((eq_qd,  0), 0.22, color='red',  alpha=0.15))
mass1  = ax_phys.add_patch(plt.Circle((eq_cav, 0), 0.25, color='#00b8ff', ec='black', zorder=10))
mass2  = ax_phys.add_patch(plt.Circle((eq_qd,  0), 0.25, color='#ff5e5e', ec='black', zorder=10))

text_cav = ax_phys.text(eq_cav, 0.50, 'Photon\n(Cavity)', ha='center',
                         fontsize=11, fontweight='bold', color='#0082b3')
text_qd  = ax_phys.text(eq_qd,  0.50, 'Exciton\n(QD)',   ha='center',
                         fontsize=11, fontweight='bold', color='#b30000')

ax_phys.text((mur_gauche + eq_cav)/2, -0.42, r'$\kappa \propto \omega_c$',
              ha='center', fontsize=10, color='black', alpha=0.7)
ax_phys.text((eq_cav + eq_qd)/2,      -0.42, r'$g(\omega_c, g_0)$',
              ha='center', fontsize=10, color='red')
ax_phys.text((eq_qd + mur_droit)/2,   -0.42, r'$\gamma \propto \omega_x$',
              ha='center', fontsize=10, color='black', alpha=0.7)

spring_cav, = ax_phys.plot([], [], color='black', lw=1,   alpha=0.7)
spring_g,   = ax_phys.plot([], [], color='red',   lw=2.0, alpha=0.9)
spring_qd,  = ax_phys.plot([], [], color='black', lw=1,   alpha=0.7)

# --- Panneau spectre — axe Y en énergie RELATIVE ω - ω_x ---
line_plus,  = ax_spec.plot(delta_range, plus_curve - omega_x,  color='red',  lw=3,
                            label=r'Upper Polariton ($\omega_+$)')
line_moins, = ax_spec.plot(delta_range, moins_curve - omega_x, color='blue', lw=3,
                            label=r'Lower Polariton ($\omega_-$)')
ax_spec.plot(delta_range, delta_range, 'k--', alpha=0.4,
             label=r'Bare Cavity ($\delta$)')
ax_spec.axhline(0, color='grey', ls='--', alpha=0.4,
                label=r'Bare QD Exciton ($\omega_x$)')
ax_spec.axvline(0, color='green', ls=':', alpha=0.4, label=r'Resonance ($\delta=0$)')

ax_spec.set_ylabel(r'Energy $\omega - \omega_x$', fontsize=13)
ax_spec.set_xlabel(r'Detuning $\delta = \omega_c - \omega_x$', fontsize=13)
_margin = G0_MAX + 1.0
ax_spec.set_xlim(DELTA_MIN, DELTA_MAX)
ax_spec.set_ylim(-DELTA_MAX - _margin, DELTA_MAX + _margin)
ax_spec.legend(loc='upper left', fontsize=10)

point_plus,  = ax_spec.plot([], [], 'ro', markersize=8)
point_moins, = ax_spec.plot([], [], 'bo', markersize=8)
line_v = ax_spec.axvline(omega_x, color='black', ls='--', alpha=0.2)

freq_text = ax_spec.text(
    0.52, 0.06, '',
    transform=ax_spec.transAxes,
    fontsize=11, fontweight='bold',
    bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=0.6')
)

# ============================================================
# 6. WIDGETS
# ============================================================
ax_radio = plt.axes([0.08, 0.05, 0.22, 0.12], facecolor='#f4f4f4')
radio    = RadioButtons(ax_radio, (
    'Photon excited, QD at rest',
    r'Lower Polariton ($\omega_-$)',
    r'Upper Polariton ($\omega_+$)'
))

ax_slider_wc = plt.axes([0.38, 0.18, 0.52, 0.03])
slider_wc    = Slider(ax_slider_wc, r'Detuning $\delta$',
                      DELTA_MIN_SAFE, DELTA_MAX, valinit=0.0, valstep=0.1)

ax_slider_g0 = plt.axes([0.38, 0.10, 0.52, 0.03])
slider_g0    = Slider(ax_slider_g0, r'Coupling $g_0$',
                      G0_MIN, G0_MAX, valinit=G0_INIT, valstep=0.05,
                      color='#ffaaaa')

# ============================================================
# 7. ÉTAT PARTAGÉ
# ============================================================
state = {
    'sols_photon': sols_photon,
    'sols_minus':  sols_minus,
    'sols_plus':   sols_plus,
    'sols_unc':    sols_unc,
    'g0':          G0_INIT,
    'plus_curve':  plus_curve,
    'moins_curve': moins_curve,
    'n_frames':    n_frames_init,
    'computing':   False,   # verrou: calcul en cours?
    'pending_g0':  None,    # dernière valeur demandée pendant le calcul
}

data_map_keys = {
    'Photon excited, QD at rest':    'sols_photon',
    r'Lower Polariton ($\omega_-$)': 'sols_minus',
    r'Upper Polariton ($\omega_+$)': 'sols_plus',
}

# ============================================================
# 8. CALLBACK g0 — calcul dans un thread séparé (plus de lag)
# ============================================================
def _do_compute(g0_new):
    """Calcul lourd dans un thread — ne touche pas matplotlib."""
    new_plus, new_moins = compute_spectrum(g0_new)
    sp, sm, spl, su, nf = compute_simulations(g0_new)

    # Écriture atomique dans state (GIL protège les affectations simples)
    state['plus_curve']  = new_plus
    state['moins_curve'] = new_moins
    state['sols_photon'] = sp
    state['sols_minus']  = sm
    state['sols_plus']   = spl
    state['sols_unc']    = su
    state['g0']          = g0_new
    state['n_frames']    = nf
    state['computing']   = False

    # Si le slider a bougé pendant le calcul, relancer avec la dernière valeur
    pending = state['pending_g0']
    if pending is not None and pending != g0_new:
        state['pending_g0'] = None
        _launch_compute(pending)
    else:
        state['pending_g0'] = None

    # Redémarrer l'animation avec le bon n_frames
    ani.frame_seq  = iter(range(nf))
    ani.save_count = nf

def _launch_compute(g0_new):
    state['computing'] = True
    t = threading.Thread(target=_do_compute, args=(g0_new,), daemon=True)
    t.start()

def on_g0_change(val):
    g0_new = slider_g0.val
    if state['computing']:
        # Calcul déjà en cours: mémoriser, sera relancé à la fin
        state['pending_g0'] = g0_new
        return
    _launch_compute(g0_new)

slider_g0.on_changed(on_g0_change)

# ============================================================
# 9. ANIMATION
# ============================================================
def update(frame):
    g0    = state['g0']
    nf    = state['n_frames']
    frame = min(frame, nf - 1)

    idx_seq       = np.argmin(np.abs(delta_targets - slider_wc.val))
    current_delta = delta_targets[idx_seq]
    current_wc    = omega_x + current_delta
    current_g     = g_of_wc(current_wc, g0)

    mode    = radio.value_selected
    sol_key = data_map_keys[mode]
    sol_c   = state[sol_key][idx_seq]
    sol_u   = state['sols_unc'][idx_seq]

    line_plus.set_xdata(delta_range)
    line_plus.set_ydata(state['plus_curve'] - omega_x)
    line_moins.set_xdata(delta_range)
    line_moins.set_ydata(state['moins_curve'] - omega_x)

    display_mode   = "Photon excited, QD at rest" if "Photon" in mode else mode
    computing_str  = "  [computing...]" if state['computing'] else ""
    ax_phys.set_title(
        rf"Mode: {display_mode}  |  $\delta={current_delta:.1f}$  ($\omega_c={current_wc:.1f}$)  |  $g_0={g0:.2f}${computing_str}",
        fontsize=13, pad=12
    )

    x_margin = 0.4
    x1_c = np.clip(eq_cav + sol_c[frame, 0], mur_gauche + x_margin, mur_droit - x_margin)
    x2_c = np.clip(eq_qd  + sol_c[frame, 1], mur_gauche + x_margin, mur_droit - x_margin)
    x1_u = np.clip(eq_cav + sol_u[frame, 0], mur_gauche + x_margin, mur_droit - x_margin)
    x2_u = np.clip(eq_qd  + sol_u[frame, 1], mur_gauche + x_margin, mur_droit - x_margin)

    mass1.center  = (x1_c, 0);  mass2.center  = (x2_c, 0)
    ghost1.center = (x1_u, 0);  ghost2.center = (x2_u, 0)
    text_cav.set_position((x1_c, 0.50))
    text_qd.set_position( (x2_c, 0.50))

    G_MAX_NOW = g_of_wc(WC_MAX, g0)
    amp_kappa = normalize_amp(current_wc, WC_MAX)
    amp_g     = normalize_amp(current_g,  max(G_MAX_NOW, 1e-6))
    amp_gamma = normalize_amp(omega_x,    WC_MAX)

    s1x, s1y = get_spring(mur_gauche,   x1_c - 0.25, amplitude=amp_kappa)
    scx, scy = get_spring(x1_c + 0.25, x2_c - 0.25, amplitude=amp_g)
    s2x, s2y = get_spring(x2_c + 0.25, mur_droit,   amplitude=amp_gamma)

    spring_cav.set_data(s1x, s1y)
    spring_g.set_data(  scx, scy)
    spring_qd.set_data( s2x, s2y)

    H = np.array([[current_wc, current_g], [current_g, omega_x]])
    w_m, w_p = np.sort(np.linalg.eigvals(H))

    # Points en énergie relative
    point_plus.set_data( [current_delta], [w_p - omega_x])
    point_moins.set_data([current_delta], [w_m - omega_x])
    line_v.set_xdata([current_delta])

    # Condition couplage fort avec κ et γ réalistes (fractions de g0, ratio κ/γ≈4 comme article)
    kappa = g0 * 0.46   # κ/g ≈ 0.46 dans l'article (52/113)
    gamma = g0 * 0.12   # γ/g ≈ 0.12 dans l'article (14/113)
    threshold = (kappa + gamma) / 2
    split  = w_p - w_m
    regime = "strong" if current_g > threshold else "weak"
    sign   = ">" if current_g > threshold else "<"

    freq_text.set_text(
        rf"$\omega_-$={w_m:.2f}  $\omega_+$={w_p:.2f}  $\delta$={current_delta:.2f}" + "\n" +
        rf"Splitting={split:.2f}  2g={2*current_g:.2f}" + "\n" +
        rf"Coherence  $\frac{{g}}{{\kappa}}$={current_g/kappa:.2f}" + "\n" +
        rf"Regime: {regime}  g={current_g:.2f} {sign} $\frac{{\kappa + \gamma}}{{2}}$={threshold:.2f}"
    )

    return (mass1, mass2, ghost1, ghost2,
            spring_cav, spring_g, spring_qd,
            line_plus, line_moins,
            point_plus, point_moins, line_v,
            freq_text, text_cav, text_qd)

ani = FuncAnimation(fig, update, frames=iter(range(n_frames_init)),
                    interval=25, blit=False,
                    save_count=300, cache_frame_data=False)
plt.show()