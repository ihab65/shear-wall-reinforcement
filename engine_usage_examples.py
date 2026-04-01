"""
engine_usage_examples.py
========================
Snippets showing every new feature added to engine.py.
Copy the relevant blocks into your Jupyter notebook.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Standard imports (same as before)
# ─────────────────────────────────────────────────────────────────────────────
from engine import (
    WallGeometry, Materials, LoadCombo,
    RebarConfig, ConfinedArea,          # ← ConfinedArea is new
    run_all, plot_stress_strain,        # ← plot_stress_strain is new
    run_ductility, make_ductility_figure,  # ← both new
    STANDARD_DIAMETERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ConfinedArea-based RebarConfig
#     Mirrors the ETABS cross-section editor (Image 1):
#       Width, w  = 400 mm   (along lw direction)
#       Height, h = 200 mm   (along bw direction)
#       Corner    = d25
#       Along 2   = 2 × d25  (intermediate bars on the two w-faces)
#       Along 3   = 2 × d25  (intermediate bars on the two h-faces)
#       Clear Cover = 38.1 mm  ← passed to WallGeometry.cover
# ─────────────────────────────────────────────────────────────────────────────

# The ConfinedArea mirrors ETABS inputs exactly.
be_layout = ConfinedArea(
    w          = 400,   # mm — Width in ETABS
    h          = 200,   # mm — Height in ETABS
    corner_dia = 25,    # mm — Corner bar Ø
    along_2    = 2,     # additional bars along each w-face (ETABS "Along 2")
    along_3    = 2,     # additional bars along each h-face (ETABS "Along 3")
    along_dia  = 25,    # mm — bar Ø for all intermediate bars
)

# RebarConfig now accepts a `confined` parameter.
# boundary_dia / boundary_n are not needed and can be omitted (they default to 0).
configs = [
    RebarConfig(
        name        = 'C1 — Critique',
        confined    = be_layout,         # ← ConfinedArea replaces boundary_dia/n
        web_dia     = 12,
        web_spacing = 150,
    ),
    # You can still use the old syntax for configs where ConfinedArea is not needed:
    RebarConfig(
        name         = 'C2 — Transition',
        boundary_dia = 20,
        boundary_n   = 4,
        web_dia      = 12,
        web_spacing  = 200,
    ),
]

# Different ConfinedArea per config:
configs_varied = [
    RebarConfig(
        name     = 'Zone critique',
        confined = ConfinedArea(w=400, h=200, corner_dia=25,
                                along_2=2, along_3=2, along_dia=25),
        web_dia=12, web_spacing=150,
    ),
    RebarConfig(
        name     = 'Zone courante',
        confined = ConfinedArea(w=350, h=200, corner_dia=20,
                                along_2=1, along_3=1, along_dia=20),
        web_dia=10, web_spacing=200,
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Stress–strain plot (new)
#     Call this anywhere in the notebook to visualise the σ-ε laws used.
# ─────────────────────────────────────────────────────────────────────────────
mat = Materials(fc28=30.0, fyk=500.0)

fig_ss = plot_stress_strain(mat)
# fig_ss.savefig("stress_strain_C30_FeE500.pdf", bbox_inches="tight")
# In Jupyter: the figure renders inline automatically (plt.show() not needed
# when using %matplotlib inline).


# ─────────────────────────────────────────────────────────────────────────────
# 3.  §7.7.5 Ductility check — standalone call
# ─────────────────────────────────────────────────────────────────────────────
geom = WallGeometry(
    lw=2.35, bw=0.20, hw=33.44, he=3.04, cover=38.1,
)
combo_gov = LoadCombo('ELU-Ex+', N_ed=-2700.0, M_base=1222.99, M_top=355.07)
cfg_base  = RebarConfig(
    name='C1',
    confined=ConfinedArea(w=400, h=200, corner_dia=25,
                          along_2=2, along_3=2, along_dia=25),
    web_dia=12, web_spacing=150,
)

dr = run_ductility(
    geom       = geom,
    mat        = mat,
    combo      = combo_gov,
    cfg_base   = cfg_base,
    T0         = 0.65,     # fundamental period T0 [s]
    T2         = 0.50,     # corner period T2 [s]
    R          = 3.0,      # behaviour coefficient R (RPA 2024)
    QF         = 1.25,     # overstrength factor Q_F
    M_RD       = 1500.0,   # design flexural resistance at base [kN·m]
    M_ED       = None,     # None → uses |combo.M_base| automatically
)

# Access individual results:
print(f"μ_φ          = {dr.mu_phi:.3f}")
print(f"α·ω_wd req.  = {dr.omega_wd_req:.4f}")
print(f"α·ω_wd prov. = {dr.omega_wd_prov:.4f}  {'✅' if dr.omega_wd_ok else '❌'}")
print(f"ε_cu,c       = {dr.eps_cu_c:.5f}  (eq. 7.38)")
print(f"x_u          = {dr.x_u:.1f} mm       (eq. 7.37)")
print(f"l_c,calcul   = {dr.lc_calcul:.1f} mm  (eq. 7.39)")
print(f"l_c,min      = {dr.lc_min:.1f} mm")
print(f"l_c,fourni   = {dr.lc_provided:.1f} mm  {'✅' if dr.lc_ok else '❌'}")

# Plot the ductility summary figure:
fig_duct = make_ductility_figure(dr, geom)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  run_all() — full pipeline including ductility (new ductility_params)
# ─────────────────────────────────────────────────────────────────────────────
combos = [
    LoadCombo('ELU-Ex+', N_ed=-2700.0, M_base=1222.99, M_top=355.07),
]
storey_map = {1: 'C1 — Critique', 4: 'C2 — Transition'}

results = run_all(
    geom       = geom,
    mat        = mat,
    combos     = combos,
    configs    = configs,
    storey_map = storey_map,
    nm_points  = 25,
    # Pass this dict to trigger §7.7.5 automatically:
    ductility_params = dict(
        T0   = 0.65,
        T2   = 0.50,
        R    = 3.0,
        QF   = 1.25,
        M_RD = 1500.0,
        # M_ED = ...  # omit to use M_base from governing combo
    ),
)

# Results are on the RunResults object as before, plus two new fields:
print(results.ductility)        # DuctilityResult dataclass
results.ductility_fig           # matplotlib Figure — renders inline in Jupyter

# All existing attributes still work:
results.nm_fig
results.section_figs['C1 — Critique']


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Accessing DuctilityResult fields for your LaTeX report
# ─────────────────────────────────────────────────────────────────────────────
dr = results.ductility
if dr is not None:
    print(
        f"\\nu_d        = {dr.nu_d:.4f}\n"
        f"\\omega_v     = {dr.omega_v:.4f}\n"
        f"\\mu_\\phi    = {dr.mu_phi:.2f}\n"
        f"\\alpha       = {dr.alpha:.4f}  (= {dr.alpha_n:.4f} × {dr.alpha_s:.4f})\n"
        f"\\omega_wd    = {dr.omega_wd_req:.4f}  (required) vs "
        f"{dr.omega_wd_prov:.4f} (provided)\n"
        f"\\varepsilon_cu,c = {dr.eps_cu_c:.6f}\n"
        f"x_u          = {dr.x_u:.1f} mm\n"
        f"l_c,calcul   = {dr.lc_calcul:.1f} mm\n"
        f"l_c,fourni   = {dr.lc_provided:.1f} mm\n"
    )
