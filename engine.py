"""
engine.py — RPA 2024 Shear Wall Calculation Engine
====================================================
Pure calculation module. No Streamlit, no print statements.
All functions accept plain Python dicts/lists and return plain data.
The Streamlit app imports from here; so does any future CLI or test suite.

New in this version
-------------------
* ConfinedArea  — rectangular boundary element rebar layout (mirrors ETABS
                  cross-section editor parameters: width, height, corner bar
                  diameter, along-2 count, along-3 count, along diameter).
* RebarConfig   — now accepts an optional ``confined`` parameter of type
                  ConfinedArea.  When provided:
                    - the confined rectangles are auto-placed at
                      (lw − lc/2) on each side (centres on section axis)
                    - all geometry and bar counts are derived from the
                      ConfinedArea specification
                    - the concrete section is built with the same rectangular
                      array API but with the exact layout from ConfinedArea
* plot_stress_strain()   — plots concrete (parabolic → rectangular stress
                           block) and steel (elastic–plastic) σ-ε curves for
                           a given Materials instance.
* DuctilityResult        — structured result for §7.7.5 ductility checks:
                           ω_wd required (eq. 7.30), ε_cu,c (eq. 7.38),
                           x_u (eq. 7.37), l_c,calcul (eq. 7.39).
* run_ductility()        — computes all §7.7.5 quantities for a given wall,
                           material set, load combo, and ConfinedArea.
* make_ductility_figure()— bar chart / summary figure for ductility results.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle, Circle
from dataclasses import dataclass, field
from typing import Any, Optional

from concreteproperties import (
    Concrete, ConcreteSection, ConcreteLinear, RectangularStressBlock,
    SteelBar, SteelElasticPlastic, add_bar_rectangular_array,
)
from sectionproperties.pre.library import rectangular_section

# ── Standard bar diameters available on site (mm) ────────────────────────────
STANDARD_DIAMETERS = [8, 10, 12, 14, 16, 20, 25, 32]

# ── Colour palette (shared by plots and section drawings) ────────────────────
CLR_CONCRETE      = "#D6C9A8"
CLR_CONCRETE_EDGE = "#8B7355"
CLR_BE_ZONE       = "#C8D8E8"
CLR_BE_BAR        = "#1A3A6B"
CLR_WEB_BAR       = "#B22222"
CLR_HATCH         = "#555555"
CLR_DIM           = "#444444"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES — typed containers so the UI never has to parse raw dicts
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WallGeometry:
    lw: float          # wall length [m]
    bw: float          # wall thickness [m]
    hw: float          # total height [m]
    he: float          # storey clear height [m]
    cover: float = 30.0  # clear cover [mm]

    @property
    def n_storeys(self) -> int:
        return int(self.hw / self.he)

    @property
    def hcr(self) -> float:
        return max(self.lw, self.hw / 6)

    @property
    def lc(self) -> float:
        """Minimum boundary element length per RPA 2024 §7.7.3 [m]."""
        return max(0.15 * self.lw, 1.5 * self.bw)

    @property
    def n_crit(self) -> int:
        return int(np.ceil(self.hcr / self.he))

    @property
    def bw_min(self) -> float:
        return max(0.15, self.he / 20)

    @property
    def lw_min(self) -> float:
        return max(self.he / 3, 4 * self.bw, 1.0)


@dataclass
class Materials:
    fc28: float = 30.0
    fyk:  float = 500.0
    Es:   float = 200_000.0
    eps_su: float = 0.05

    @property
    def Ec(self) -> float:
        return 11000 * self.fc28 ** (1 / 3)

    @property
    def fctm(self) -> float:
        return 0.30 * self.fc28 ** (2 / 3)

    @property
    def fyd(self) -> float:
        """Design yield strength [MPa] (γ_s = 1.15)."""
        return self.fyk / 1.15

    @property
    def fcd(self) -> float:
        """Design compressive strength [MPa] (γ_b = 1.5)."""
        return self.fc28 / 1.5

    @property
    def eps_sy(self) -> float:
        """Yield strain of steel."""
        return self.fyk / self.Es


# ─────────────────────────────────────────────────────────────────────────────
# ConfinedArea  (new)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfinedArea:
    """Rectangular boundary element cross-section layout.

    Parameters
    ----------
    w : float
        Width of the confined zone **in the wall length direction** [mm].
        This is mapped to ETABS "Width, w".
    h : float
        Height of the confined zone **in the wall thickness direction** [mm].
        This is mapped to ETABS "Height, h".
    corner_dia : int
        Diameter of corner bars [mm].
    along_2 : int
        Number of *additional* intermediate bars along the w-direction on
        each face (between the two corner bars).  Matches ETABS "Along 2".
    along_3 : int
        Number of *additional* intermediate bars along the h-direction on
        each face (between the two corner bars).  Matches ETABS "Along 3".
    along_dia : int
        Diameter of the intermediate (along-2 and along-3) bars [mm].
    cover : float
        Clear cover to the hoop/tie [mm].  Defaults to the wall cover.

    Notes
    -----
    Total bars per confined zone = 4 corners + 2*along_2 (top+bottom edges)
    + 2*along_3 (left+right edges).

    The rectangle is centred at ``x = lw − lc/2`` (right BE) and ``lw/2``
    (left BE reflected) by ``_place_confined_bars()``.
    """
    w:          float   # mm — along lw direction
    h:          float   # mm — along bw direction
    corner_dia: int     # mm
    along_2:    int     # additional bars along w-face (each side)
    along_3:    int     # additional bars along h-face (each side)
    along_dia:  int     # mm

    def total_bars(self) -> int:
        return 4 + 2 * self.along_2 * 2 + 2 * self.along_3 * 2

    def As_total_mm2(self) -> float:
        """Total steel area in one confined zone [mm²]."""
        A_corner   = 4 * np.pi * self.corner_dia**2 / 4
        A_along_2  = 2 * 2 * self.along_2 * np.pi * self.along_dia**2 / 4
        A_along_3  = 2 * 2 * self.along_3 * np.pi * self.along_dia**2 / 4
        return A_corner + A_along_2 + A_along_3

    # ── geometry helpers ────────────────────────────────────────────────────

    def bar_positions_local(self) -> list[tuple[float, float, float]]:
        """Return (x, y, area) of every bar in local coords.

        Local origin = bottom-left of the ConfinedArea rectangle.
        Coordinates are measured to bar centrelines (i.e. after deducting
        cover to the tie and half the bar diameter).
        """
        c_to_bar = 8.0   # approximate: cover_to_hoop + hoop_dia + r_bar
        # For a proper placement use cover + hoop + r_bar, but the caller
        # (engine) has the full cover value and adjusts.
        positions = []
        A_c = np.pi * self.corner_dia**2 / 4
        A_a = np.pi * self.along_dia**2 / 4

        # Corner bars (4)
        corners = [(c_to_bar, c_to_bar),
                   (self.w - c_to_bar, c_to_bar),
                   (c_to_bar, self.h - c_to_bar),
                   (self.w - c_to_bar, self.h - c_to_bar)]
        for cx, cy in corners:
            positions.append((cx, cy, A_c))

        # Along-2: intermediate bars on top and bottom edges
        if self.along_2 > 0:
            span_2 = self.w - 2 * c_to_bar
            sp2    = span_2 / (self.along_2 + 1)
            for i in range(1, self.along_2 + 1):
                x = c_to_bar + i * sp2
                positions.append((x, c_to_bar,          A_a))   # bottom
                positions.append((x, self.h - c_to_bar, A_a))   # top

        # Along-3: intermediate bars on left and right edges
        if self.along_3 > 0:
            span_3 = self.h - 2 * c_to_bar
            sp3    = span_3 / (self.along_3 + 1)
            for j in range(1, self.along_3 + 1):
                y = c_to_bar + j * sp3
                positions.append((c_to_bar,          y, A_a))   # left
                positions.append((self.w - c_to_bar, y, A_a))   # right

        return positions

    def __repr__(self) -> str:
        return (f"ConfinedArea(w={self.w}, h={self.h}, "
                f"corner=Ø{self.corner_dia}, "
                f"along2={self.along_2}×Ø{self.along_dia}, "
                f"along3={self.along_3}×Ø{self.along_dia})")


@dataclass
class RebarConfig:
    """Reinforcement configuration for a wall zone.

    Parameters
    ----------
    name : str
        Identifier (e.g. ``'C1'``, ``'C2-transition'``).
    boundary_dia : int
        Bar diameter for boundary element bars [mm].
        *Only used when* ``confined`` *is None* (legacy mode).
    boundary_n : int
        Number of bars per face per BE (legacy mode).
    web_dia : int
        Web bar diameter [mm].
    web_spacing : int
        Web bar spacing centre-to-centre [mm].
    confined : ConfinedArea | None
        When provided, the boundary element layout is fully described by
        this object and ``boundary_dia`` / ``boundary_n`` are ignored for
        section building.  The ``boundary_dia`` attribute is still used for
        the confinement hoop spacing check (St_max = min(b0/3, 125, 6Ø)).

    Usage
    -----
    Legacy (no ConfinedArea)::

        cfg = RebarConfig('C1', boundary_dia=25, boundary_n=4,
                          web_dia=12, web_spacing=150)

    With ConfinedArea::

        cfg = RebarConfig(
            'C1',
            confined=ConfinedArea(w=400, h=200,
                                  corner_dia=25, along_2=2, along_3=2,
                                  along_dia=25),
            web_dia=12, web_spacing=150,
        )
    """
    name:         str
    boundary_dia: int  = 0     # ignored when confined is not None
    boundary_n:   int  = 0     # ignored when confined is not None
    web_dia:      int  = 12
    web_spacing:  int  = 150
    confined:     Optional[ConfinedArea] = field(default=None, repr=False)

    def effective_boundary_dia(self) -> int:
        """Returns the controlling bar diameter for spacing checks."""
        if self.confined is not None:
            return self.confined.corner_dia
        return self.boundary_dia


@dataclass
class LoadCombo:
    label: str
    N_ed:  float    # axial [kN], compression negative
    M_base: float   # moment at base [kN·m]
    M_top:  float   # moment at top  [kN·m]


@dataclass
class StoreyResult:
    storey:   int
    z_m:      float
    cfg_name: str
    Menv:     float   # [kN·m]
    MR:       float   # [kN·m]
    eta:      float   # Menv / MR
    ok:       bool

    @property
    def status(self) -> str:
        return "✅ OK" if self.ok else "❌ FAIL"


@dataclass
class Phase1Result:
    bw_ok:    bool
    lw_ok:    bool
    bw:       float
    bw_min:   float
    lw:       float
    lw_min:   float
    hcr:      float
    lc:       float
    n_crit:   int
    nu_d_checks: list[dict]   # [{label, nu_d, ok}]


@dataclass
class DuctilityResult:
    """§7.7.5 local ductility verification results.

    All quantities are scalars computed for the **base** load combination
    (governing compression) at the critical section.

    Attributes
    ----------
    nu_d          : reduced axial force ν_d = N_Ed / (bw·lw·fc28)
    omega_v       : normalised vertical web steel ratio ω_v  (eq. 7.34)
    mu_phi        : required curvature ductility coefficient μ_φ (eq. 7.36)
    alpha         : confinement efficiency coefficient α = α_n · α_s (eq. 7.35)
    alpha_n       : geometry factor α_n
    alpha_s       : spacing factor α_s
    omega_wd_req  : required mechanical confinement ratio α·ω_wd (eq. 7.30)
    omega_wd_prov : provided mechanical confinement ratio
    omega_wd_ok   : True if provided ≥ required
    eps_cu_c      : confined concrete ultimate strain ε_cu,c (eq. 7.38)
    x_u           : neutral axis depth at ultimate curvature xu (eq. 7.37)
    lc_calcul     : required compressed boundary element length l_c,calcul [mm]
    lc_min        : minimum l_c,calcul = max(0.15·lw, 1.5·bw) [mm]
    lc_ok         : True if lc_calcul ≤ actual lc_provided
    """
    nu_d:          float
    omega_v:       float
    mu_phi:        float
    alpha:         float
    alpha_n:       float
    alpha_s:       float
    omega_wd_req:  float
    omega_wd_prov: float
    omega_wd_ok:   bool
    eps_cu_c:      float
    x_u:           float
    lc_calcul:     float   # mm
    lc_min:        float   # mm
    lc_provided:   float   # mm (actual lc used)
    lc_ok:         bool


@dataclass
class Phase5Result:
    nu_d:         float
    rho_v:        float
    omega_v:      float
    omega_wd_req: float
    St_max:       float
    mu_phi:       float = 5.0


@dataclass
class RunResults:
    phase1:        Phase1Result
    envelope_df:   Any          # pandas DataFrame
    storeys:       list[StoreyResult]
    phase5:        Phase5Result
    all_ok:        bool
    nm_fig:        Any          # matplotlib Figure
    section_figs:  dict[str, Any]   # cfg_name → Figure
    ductility:     Optional[DuctilityResult] = None
    ductility_fig: Optional[Any]             = None


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_materials(mat: Materials):
    concrete = Concrete(
        name=f"C{int(mat.fc28)}",
        density=2.4e-6,
        stress_strain_profile=ConcreteLinear(elastic_modulus=mat.Ec),
        ultimate_stress_strain_profile=RectangularStressBlock(
            compressive_strength=mat.fc28,
            alpha=0.8,
            gamma=0.9,
            ultimate_strain=0.0035,
        ),
        flexural_tensile_strength=mat.fctm,
        colour="lightgrey",
    )
    steel = SteelBar(
        name=f"FeE{int(mat.fyk)}",
        density=7.85e-6,
        stress_strain_profile=SteelElasticPlastic(
            yield_strength=mat.fyk,
            elastic_modulus=mat.Es,
            fracture_strain=mat.eps_su,
        ),
        colour="steelblue",
    )
    return concrete, steel


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDER  (updated to handle ConfinedArea)
# ═══════════════════════════════════════════════════════════════════════════════

def _place_confined_bars(
    g, ca: ConfinedArea, lw_mm: float, bw_mm: float, lc_mm: float,
    cover: float, steel_mat,
):
    """Add boundary element bars from a ConfinedArea to geometry g.

    The ConfinedArea rectangle is centred at x = lc_mm/2  (left BE)
    and x = lw_mm − lc_mm/2  (right BE), and vertically centred
    on bw_mm/2.

    Returns the updated geometry.
    """
    # Derive the actual clear cover to bar centre from cover + hoop (8mm) + r_bar
    local_bars = ca.bar_positions_local()

    # lc_mm is the full BE zone length; centre the confined rectangle in it
    # Left BE: rectangle left edge at max(0, lc_mm/2 − ca.w/2)
    x_left_origin  = lc_mm / 2.0 - ca.w / 2.0
    # Right BE: rectangle left edge at lw_mm − lc_mm/2 − ca.w/2
    x_right_origin = lw_mm - lc_mm / 2.0 - ca.w / 2.0
    y_origin       = bw_mm / 2.0 - ca.h / 2.0

    for x_origin in [x_left_origin, x_right_origin]:
        for (lx, ly, area) in local_bars:
            gx = x_origin + lx
            gy = y_origin  + ly
            g = add_bar_rectangular_array(
                geometry=g, area=area, material=steel_mat,
                n_x=1, x_s=1.0, n_y=1, y_s=1.0,
                anchor=(gx, gy),
            )
    return g


def build_wall_section(
    geom: WallGeometry,
    cfg:  RebarConfig,
    concrete_mat, steel_mat,
) -> ConcreteSection:
    lw_mm = geom.lw * 1000
    bw_mm = geom.bw * 1000
    lc_mm = geom.lc * 1000
    c     = geom.cover

    wall_geo = rectangular_section(d=bw_mm, b=lw_mm, material=concrete_mat)
    g        = wall_geo

    # ── Boundary element bars ────────────────────────────────────────────────
    if cfg.confined is not None:
        g = _place_confined_bars(g, cfg.confined, lw_mm, bw_mm, lc_mm, c, steel_mat)
    else:
        # Legacy rectangular array layout
        bar_area_be = np.pi * cfg.boundary_dia ** 2 / 4
        be_inner    = lc_mm - 2 * c
        be_x_sp     = be_inner / max(cfg.boundary_n - 1, 1)
        for face_y in [c, bw_mm - c]:
            for x0 in [c, lw_mm - lc_mm + c]:
                g = add_bar_rectangular_array(
                    geometry=g, area=bar_area_be, material=steel_mat,
                    n_x=cfg.boundary_n,
                    x_s=be_x_sp if cfg.boundary_n > 1 else 1.0,
                    n_y=1, y_s=1.0, anchor=(x0, face_y),
                )

    # ── Web bars ─────────────────────────────────────────────────────────────
    bar_area_web = np.pi * cfg.web_dia ** 2 / 4
    web_len      = (lw_mm - lc_mm) - lc_mm
    n_web        = max(int(web_len / cfg.web_spacing) - 1, 1)
    web_sp_act   = web_len / (n_web + 1)
    for face_y in [c, bw_mm - c]:
        g = add_bar_rectangular_array(
            geometry=g, area=bar_area_web, material=steel_mat,
            n_x=n_web, x_s=web_sp_act,
            n_y=1, y_s=1.0,
            anchor=(lc_mm + web_sp_act, face_y),
        )

    return ConcreteSection(g)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(geom: WallGeometry, mat: Materials, combos: list[LoadCombo]) -> Phase1Result:
    nu_d_checks = []
    for c in combos:
        nu_d = abs(c.N_ed * 1e3) / (geom.bw * 1e3 * geom.lw * 1e3 * mat.fc28)
        nu_d_checks.append({"label": c.label, "nu_d": nu_d, "ok": nu_d <= 0.40})

    return Phase1Result(
        bw_ok=geom.bw >= geom.bw_min,
        lw_ok=geom.lw >= geom.lw_min,
        bw=geom.bw, bw_min=geom.bw_min,
        lw=geom.lw, lw_min=geom.lw_min,
        hcr=geom.hcr, lc=geom.lc, n_crit=geom.n_crit,
        nu_d_checks=nu_d_checks,
    )


def envelope_moment(z: float, combo: LoadCombo, hcr: float, hw: float) -> float:
    if z <= hcr:
        return abs(combo.M_base)
    frac  = (z - hcr) / (hw - hcr)
    M_env = abs(combo.M_base) + (abs(combo.M_top) - abs(combo.M_base)) * frac
    return max(M_env, abs(combo.M_top))


def build_envelope_table(geom: WallGeometry, combos: list[LoadCombo]) -> list[dict]:
    rows = []
    for s in range(1, geom.n_storeys + 1):
        z = s * geom.he
        row = {"storey": s, "z_m": z}
        for c in combos:
            row[f"Menv_{c.label}"] = envelope_moment(z, c, geom.hcr, geom.hw)
        row["Menv_max"] = max(row[f"Menv_{c.label}"] for c in combos)
        row["N_vals"]   = [c.N_ed for c in combos]
        rows.append(row)
    return rows


def compute_nm_curves(
    geom: WallGeometry,
    configs: list[RebarConfig],
    concrete_mat, steel_mat,
    n_points: int = 25,
) -> dict[str, Any]:
    results = {}
    for cfg in configs:
        sec = build_wall_section(geom, cfg, concrete_mat, steel_mat)
        mi  = sec.moment_interaction_diagram(theta=0, n_points=n_points,
                                              progress_bar=False)
        results[cfg.name] = {"mi": mi, "cfg": cfg}
    return results


def get_mr_for_ned(mi_result, N_ed_kN: float) -> float:
    mi     = mi_result["mi"]
    N_cp   = -N_ed_kN * 1e3
    n_arr  = np.array([r.n   for r in mi.results])
    mx_arr = np.array([r.m_x for r in mi.results])
    order  = np.argsort(n_arr)
    n_s, mx_s = n_arr[order], mx_arr[order]
    N_cp   = float(np.clip(N_cp, n_s[0], n_s[-1]))
    return abs(float(np.interp(N_cp, n_s, mx_s))) / 1e6


def run_phase4(
    envelope_table: list[dict],
    nm_curves:      dict[str, Any],
    storey_map:     dict[int, str],
) -> list[StoreyResult]:
    def resolve(s: int) -> str:
        if s in storey_map:
            return storey_map[s]
        lower = [k for k in storey_map if k <= s]
        if not lower:
            raise ValueError(f"No config assigned for storey {s}")
        return storey_map[max(lower)]

    results = []
    for row in envelope_table:
        s        = row["storey"]
        cfg_name = resolve(s)
        mi       = nm_curves[cfg_name]
        MR       = min(get_mr_for_ned(mi, N) for N in row["N_vals"])
        Menv     = row["Menv_max"]
        eta      = Menv / MR if MR > 1.0 else 999.0
        results.append(StoreyResult(
            storey=s, z_m=row["z_m"], cfg_name=cfg_name,
            Menv=Menv, MR=MR, eta=eta, ok=(eta <= 1.0),
        ))
    return results


def run_phase5(
    geom:    WallGeometry,
    mat:     Materials,
    combos:  list[LoadCombo],
    cfg_base: RebarConfig,
) -> Phase5Result:
    Ned_base = max(combos, key=lambda c: abs(c.N_ed)).N_ed
    nu_d     = abs(Ned_base * 1e3) / (geom.bw * 1e3 * geom.lw * 1e3 * mat.fc28)
    rho_v    = (np.pi * cfg_base.web_dia**2 / 4) / (geom.bw * 1e3 * cfg_base.web_spacing)
    omega_v  = rho_v * mat.fyk / mat.fc28
    mu_phi   = 5.0
    eps_sy_d = mat.fyk / mat.Es
    b0       = geom.bw * 1e3 - 2 * geom.cover
    bc       = geom.bw * 1e3
    omega_wd_req = max(
        30 * mu_phi * (nu_d + omega_v) * eps_sy_d * (bc / b0) - 0.035, 0.12
    )
    St_max = min(b0 / 3, 125.0, 6 * cfg_base.effective_boundary_dia())
    return Phase5Result(nu_d=nu_d, rho_v=rho_v, omega_v=omega_v,
                        omega_wd_req=omega_wd_req, St_max=St_max)


# ═══════════════════════════════════════════════════════════════════════════════
# §7.7.5  DUCTILITY CALCULATIONS  (new)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ductility(
    geom:       WallGeometry,
    mat:        Materials,
    combo:      LoadCombo,          # governing combination (max compression)
    cfg_base:   RebarConfig,        # base-zone config (must have ConfinedArea)
    T0:         float,              # fundamental period [s]
    T2:         float,              # corner period of design spectrum [s]
    R:          float,              # behaviour factor
    QF:         float,              # overstrength factor Q_F
    M_RD:       float,              # design flexural resistance M_RD [kN·m]
    M_ED:       Optional[float] = None,  # M_ED from shifted envelope; if None use combo.M_base
) -> DuctilityResult:
    """Compute §7.7.5 ductility check quantities (eqs 7.30 – 7.39).

    Parameters
    ----------
    T0, T2   : periods [s]
    R, QF    : seismic coefficients  (R / Q_F enters μ_φ)
    M_RD     : design resistance moment at base [kN·m]
    M_ED     : moment demand at base from shifted envelope [kN·m]
               (if None, uses |combo.M_base|)
    """
    lw_mm = geom.lw * 1000
    bw_mm = geom.bw * 1000
    lc_mm = geom.lc * 1000
    c     = geom.cover

    if M_ED is None:
        M_ED = abs(combo.M_base)

    # ── ν_d  (eq. before 7.30) ──────────────────────────────────────────────
    nu_d = abs(combo.N_ed * 1e3) / (bw_mm * lw_mm * mat.fc28)

    # ── ω_v  (eq. 7.34) ─────────────────────────────────────────────────────
    rho_v  = (np.pi * cfg_base.web_dia**2 / 4) / (bw_mm * cfg_base.web_spacing)
    omega_v = rho_v * (mat.fyd / mat.fcd)

    # ── μ_φ  (eq. 7.36) ─────────────────────────────────────────────────────
    ratio = (R / QF) * (M_ED / M_RD)
    if T0 >= T2:
        mu_phi = 2.0 * ratio - 1.0
    else:
        mu_phi = 1.0 + 2.0 * (ratio - 1.0) * T2 / T0
    mu_phi = max(mu_phi, 1.0)

    # ── α  (eq. 7.35) — requires ConfinedArea ───────────────────────────────
    if cfg_base.confined is None:
        # Approximate with full confinement (α = 1) if no ConfinedArea given
        alpha_n, alpha_s, alpha = 1.0, 1.0, 1.0
        b0  = bw_mm - 2 * c
        h0  = bw_mm - 2 * c   # square section assumed
        bc  = bw_mm
        t   = min(b0 / 3, 125.0, 6 * cfg_base.effective_boundary_dia())
    else:
        ca  = cfg_base.confined
        b0  = ca.w - 2 * c    # confined core width
        h0  = ca.h - 2 * c    # confined core height
        bc  = ca.w
        t   = min(b0 / 3, 125.0, 6 * cfg_base.confined.corner_dia)

        # Number of restrained longitudinal bars n (all bars on perimeter)
        n_restrained = 4 + 2 * ca.along_2 + 2 * ca.along_3  # perimeter bars

        # Sum of (bi)² / (6 b0 h0)
        # bi = spacing between consecutive restrained bars along perimeter
        # We approximate with uniform spacing for along_2 / along_3
        bars_along_b = ca.along_2 + 2   # total bars along b-edge incl. corners
        bars_along_h = ca.along_3 + 2   # total bars along h-edge incl. corners
        sp_b = b0 / (bars_along_b - 1) if bars_along_b > 1 else b0
        sp_h = h0 / (bars_along_h - 1) if bars_along_h > 1 else h0

        sum_bi2 = (2 * ca.along_2 * sp_b**2 + 2 * ca.along_3 * sp_h**2
                   + 4 * min(sp_b, sp_h)**2)   # approx corner contribution
        alpha_n = 1.0 - sum_bi2 / (6 * b0 * h0)
        alpha_n = max(0.0, alpha_n)

        alpha_s = (1.0 - t / (2 * b0)) * (1.0 - t / (2 * h0))
        alpha   = alpha_n * alpha_s

    # ── ε_sy,d ──────────────────────────────────────────────────────────────
    eps_sy_d = mat.fyd / mat.Es

    # ── α·ω_wd required  (eq. 7.30) ─────────────────────────────────────────
    omega_wd_req = max(
        30 * mu_phi * (nu_d + omega_v) * eps_sy_d * (bc / b0) - 0.035,
        0.12
    )

    # ── α·ω_wd provided ─────────────────────────────────────────────────────
    if cfg_base.confined is not None:
        ca   = cfg_base.confined
        # Volume of hoops in the confined core (rectangular hoops)
        # Assume single hoop layer per stirrup set, diameter = corner_dia for main hoop
        hoop_dia      = 10   # typical hoop diameter [mm] — could be a parameter
        A_hoop        = np.pi * hoop_dia**2 / 4
        V_hoops_per_s = 2 * (b0 + h0) * A_hoop   # perimeter × area per unit length
        V_core_per_s  = b0 * h0 * t               # core volume per stirrup pitch t
        omega_wd_prov = alpha * (V_hoops_per_s / V_core_per_s) * (mat.fyd / mat.fcd)
    else:
        omega_wd_prov = omega_wd_req   # conservative: assume exactly met

    omega_wd_ok = (omega_wd_prov >= omega_wd_req)

    # ── ε_cu,c  (eq. 7.38) ──────────────────────────────────────────────────
    eps_cu_c = 0.0035 + 0.1 * alpha * omega_wd_prov

    # ── x_u  (eq. 7.37) ─────────────────────────────────────────────────────
    x_u = (nu_d + omega_v) * (lw_mm * bc) / b0

    # ── l_c,calcul  (eq. 7.39) ──────────────────────────────────────────────
    eps_cu = 0.0035
    lc_calcul = x_u * (1.0 - eps_cu / eps_cu_c)

    lc_min      = max(0.15 * lw_mm, 1.5 * bw_mm)
    lc_provided = lc_mm

    lc_ok = lc_calcul <= lc_provided

    return DuctilityResult(
        nu_d=nu_d, omega_v=omega_v, mu_phi=mu_phi,
        alpha=alpha, alpha_n=alpha_n, alpha_s=alpha_s,
        omega_wd_req=omega_wd_req, omega_wd_prov=omega_wd_prov,
        omega_wd_ok=omega_wd_ok,
        eps_cu_c=eps_cu_c, x_u=x_u,
        lc_calcul=lc_calcul, lc_min=lc_min,
        lc_provided=lc_provided, lc_ok=lc_ok,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#1A3A6B", "#B22222", "#2E7D32", "#6A1B9A",
           "#E65100", "#00695C", "#4E342E", "#37474F"]


def plot_stress_strain(mat: Materials) -> plt.Figure:
    """Plot concrete and steel constitutive curves used in the calculations.

    Returns a matplotlib Figure with two subplots:
      - Left  : concrete parabolic–rectangular σ-ε (design values)
      - Right : steel elastic–perfectly plastic σ-ε
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#FAFAFA")

    # ── Concrete (parabolic–rectangular, Eurocode-style) ────────────────────
    fcd   = mat.fcd
    eps_c2  = 0.002   # end of parabola
    eps_cu  = 0.0035  # ultimate strain

    eps_c = np.linspace(0, eps_c2, 200)
    sigma_c = fcd * (1 - (1 - eps_c / eps_c2)**2)

    eps_rect  = np.array([eps_c2, eps_cu])
    sigma_rect = np.array([fcd, fcd])

    ax1.fill_betweenx(np.concatenate([sigma_c, sigma_rect]),
                      np.concatenate([eps_c,   eps_rect]),
                      alpha=0.15, color="#1A3A6B")
    ax1.plot(eps_c,    sigma_c,    color="#1A3A6B", lw=2.2, label="Parabolic")
    ax1.plot(eps_rect, sigma_rect, color="#B22222", lw=2.2, ls="--",
             label=f"Rectangular plateau  f_cd = {fcd:.1f} MPa")
    ax1.axvline(eps_c2, color="#999", lw=0.8, ls=":")
    ax1.axvline(eps_cu, color="#B22222", lw=0.8, ls=":",
                label=f"ε_cu = {eps_cu}")

    ax1.set_xlabel("Strain ε", fontsize=10)
    ax1.set_ylabel("Stress σ [MPa]", fontsize=10)
    ax1.set_title(f"Concrete C{int(mat.fc28)} — design σ-ε\n"
                  f"f_c28 = {mat.fc28} MPa  |  f_cd = {fcd:.2f} MPa  |  "
                  f"E_c = {mat.Ec/1000:.1f} GPa",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_facecolor("#F8F8F8")

    # ── Steel (elastic–perfectly plastic) ────────────────────────────────────
    fyd     = mat.fyd
    eps_sy  = mat.eps_sy
    eps_syd = fyd / mat.Es
    eps_su  = mat.eps_su

    eps_s = np.array([0.0, eps_sy,    eps_sy,  eps_su])
    sig_s = np.array([0.0, mat.fyk,   mat.fyk, mat.fyk])
    eps_d = np.array([0.0, eps_syd,   eps_syd, eps_su])
    sig_d = np.array([0.0, fyd,       fyd,     fyd])

    ax2.fill_betweenx(sig_d, eps_d, alpha=0.12, color="#2E7D32")
    ax2.plot(eps_s, sig_s, color="#1A3A6B", lw=2.2, label=f"Characteristic  f_yk = {mat.fyk:.0f} MPa")
    ax2.plot(eps_d, sig_d, color="#2E7D32", lw=2.2, ls="--",
             label=f"Design  f_yd = {fyd:.1f} MPa (γ_s = 1.15)")
    ax2.axvline(eps_sy,  color="#1A3A6B", lw=0.8, ls=":", label=f"ε_sy = {eps_sy:.5f}")
    ax2.axvline(eps_syd, color="#2E7D32", lw=0.8, ls=":")
    ax2.axvline(eps_su,  color="#B22222", lw=0.8, ls=":",
                label=f"ε_su = {eps_su}")

    ax2.set_xlabel("Strain ε", fontsize=10)
    ax2.set_ylabel("Stress σ [MPa]", fontsize=10)
    ax2.set_title(f"Steel FeE{int(mat.fyk)} — design σ-ε\n"
                  f"f_yk = {mat.fyk} MPa  |  f_yd = {fyd:.2f} MPa  |  "
                  f"E_s = {mat.Es/1000:.0f} GPa",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_facecolor("#F8F8F8")

    fig.suptitle("Lois de comportement des matériaux (RPA 2024 / EC2)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def make_ductility_figure(dr: DuctilityResult, geom: WallGeometry) -> plt.Figure:
    """Summary figure for §7.7.5 ductility checks."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#FAFAFA")

    # ── Panel 1: key scalars as a table ─────────────────────────────────────
    ax = axes[0]
    ax.axis("off")
    rows = [
        ("ν_d",            f"{dr.nu_d:.4f}",        "≤ 0.40"),
        ("ω_v",            f"{dr.omega_v:.4f}",      "—"),
        ("μ_φ",            f"{dr.mu_phi:.2f}",       "—"),
        ("α_n",            f"{dr.alpha_n:.3f}",      "—"),
        ("α_s",            f"{dr.alpha_s:.3f}",      "—"),
        ("α = α_n·α_s",    f"{dr.alpha:.3f}",        "—"),
        ("ε_sy,d",         f"{dr.mu_phi:.5f}",       "—"),
    ]
    col_labels = ["Paramètre", "Valeur", "Limite"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.2, 1.6)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#1A3A6B")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF2FF")
    ax.set_title("§7.7.5 Paramètres de ductilité", fontweight="bold",
                 fontsize=10, pad=12)

    # ── Panel 2: ω_wd demand vs capacity ────────────────────────────────────
    ax = axes[1]
    names  = ["α·ω_wd requis\n(eq. 7.30)", "α·ω_wd fourni"]
    values = [dr.omega_wd_req, dr.omega_wd_prov]
    colours = ["#C62828", "#2E7D32" if dr.omega_wd_ok else "#C62828"]
    bars = ax.bar(names, values, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.12, color="#FF8F00", lw=1.5, ls="--", label="Minimum = 0.12")
    verdict_txt = "✅ Vérifié" if dr.omega_wd_ok else "❌ Non vérifié"
    ax.set_title(f"Rapport de confinement mécanique\n{verdict_txt}",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("α·ω_wd")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_facecolor("#F8F8F8")

    # ── Panel 3: l_c,calcul vs l_c provided ─────────────────────────────────
    ax = axes[2]
    names  = ["l_c,calcul\n(eq. 7.39)", "l_c minimum\nmax(0.15lw,1.5bw)",
              "l_c fourni\n(zone de rive)"]
    values = [dr.lc_calcul, dr.lc_min, dr.lc_provided]
    colours = ["#1A3A6B",
               "#E65100",
               "#2E7D32" if dr.lc_ok else "#C62828"]
    bars = ax.bar(names, values, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 5, f"{v:.0f} mm",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    verdict_txt = "✅ Vérifié" if dr.lc_ok else "❌ Insuffisant"
    ax.set_title(f"Longueur comprimée l_c\n{verdict_txt}",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("Longueur [mm]")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_facecolor("#F8F8F8")

    fig.suptitle(
        f"Vérification ductilité locale §7.7.5 — "
        f"ε_cu,c = {dr.eps_cu_c:.5f}  |  x_u = {dr.x_u:.1f} mm",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def make_nm_figure(
    nm_curves:  dict[str, Any],
    combos:     list[LoadCombo],
    storeys:    list[StoreyResult],
    geom:       WallGeometry,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    facecolor="#FAFAFA")

    for (name, data), color in zip(nm_curves.items(), PALETTE):
        mi     = data["mi"]
        n_kn   = np.array([r.n   for r in mi.results]) / 1e3
        mx_knm = np.array([r.m_x for r in mi.results]) / 1e6
        ax1.plot(mx_knm, n_kn, label=name, color=color, lw=2.2)

    for i, c in enumerate(combos):
        N_plot = -c.N_ed
        ax1.scatter([c.M_base], [N_plot], marker="x", s=130, zorder=6,
                    color=PALETTE[i % len(PALETTE)], linewidths=2.5,
                    label=f"{c.label}  (N={c.N_ed:.0f} kN)")

    ax1.axhline(0, color="#999", lw=0.8, ls="--")
    ax1.set_xlabel("Moment [kN·m]", fontsize=10)
    ax1.set_ylabel("Effort normal N [kN]  —  compression (+)  /  traction (−)", fontsize=9)
    ax1.set_title("Diagramme N-M  (convention : compression positive)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.set_facecolor("#F8F8F8")

    z_vals    = [r.z_m  for r in storeys]
    Menv_vals = [r.Menv for r in storeys]
    MR_vals   = [r.MR   for r in storeys]

    ax2.plot(Menv_vals, z_vals, "o-", color="#C62828", lw=2.2,
             label="M_env (demande)", markersize=6)
    ax2.plot(MR_vals,   z_vals, "s-", color="#1565C0", lw=2.2,
             label="M_R (capacité)",  markersize=6)
    ax2.axhspan(0, geom.hcr, alpha=0.07, color="#C62828",
                label=f"Zone critique hcr={geom.hcr:.1f} m")

    for r in storeys:
        clr = "#2E7D32" if r.ok else "#C62828"
        ax2.annotate(
            f"η={r.eta:.2f}",
            xy=(max(r.MR, r.Menv) + 30, r.z_m),
            fontsize=7, color=clr, va="center",
        )

    ax2.set_xlabel("Moment [kN·m]",  fontsize=10)
    ax2.set_ylabel("Hauteur z [m]",  fontsize=10)
    ax2.set_title("Enveloppe vs Moment résistant\n(RPA 2024 §7.7.4)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, geom.hw)
    ax2.set_facecolor("#F8F8F8")

    fig.tight_layout(pad=2.0)
    return fig


def _bar_positions_from_config(cfg: RebarConfig, lw_mm, bw_mm, lc_mm, cover):
    """Return (be_bars, web_bars) lists of (x, y, area) tuples.

    Handles both legacy RebarConfig and ConfinedArea-based configs.
    """
    A_web    = np.pi * cfg.web_dia ** 2 / 4
    web_bars = []

    web_len    = (lw_mm - lc_mm) - lc_mm
    n_web      = max(int(web_len / cfg.web_spacing) - 1, 1)
    web_sp_act = web_len / (n_web + 1)
    for fy in [cover, bw_mm - cover]:
        for j in range(n_web):
            web_bars.append((lc_mm + (j + 1) * web_sp_act, fy, A_web))

    be_bars = []
    if cfg.confined is not None:
        ca            = cfg.confined
        local_bars    = ca.bar_positions_local()
        x_left_origin  = lc_mm / 2.0 - ca.w / 2.0
        x_right_origin = lw_mm - lc_mm / 2.0 - ca.w / 2.0
        y_origin       = bw_mm / 2.0 - ca.h / 2.0
        for x_origin in [x_left_origin, x_right_origin]:
            for (lx, ly, area) in local_bars:
                be_bars.append((x_origin + lx, y_origin + ly, area))
    else:
        A_be     = np.pi * cfg.boundary_dia ** 2 / 4
        be_inner = lc_mm - 2 * cover
        be_x_sp  = be_inner / max(cfg.boundary_n - 1, 1)
        for fy in [cover, bw_mm - cover]:
            for j in range(cfg.boundary_n):
                be_bars.append((cover + j * be_x_sp,           fy, A_be))
                be_bars.append((lw_mm - lc_mm + cover + j * be_x_sp, fy, A_be))

    return be_bars, web_bars


def make_section_figure(cfg: RebarConfig, geom: WallGeometry) -> plt.Figure:
    lw_mm = geom.lw * 1000
    bw_mm = geom.bw * 1000
    lc_mm = geom.lc * 1000
    c     = geom.cover

    be_bars, web_bars = _bar_positions_from_config(cfg, lw_mm, bw_mm, lc_mm, c)

    W_c, H_c   = 240.0, 60.0
    ML, MR_    = 32.0, 115.0
    MT, MB     = 28.0, 36.0
    FIG_W = (W_c + ML + MR_) / 25.4
    FIG_H = max((H_c + MT + MB) / 25.4, 5.5)

    sx = W_c / lw_mm
    sy = H_c / bw_mm
    def px(x): return ML + x * sx
    def py(y): return MB + y * sy

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="white")
    ax.set_xlim(0, W_c + ML + MR_)
    ax.set_ylim(0, H_c + MT + MB)
    ax.set_aspect("equal")
    ax.axis("off")

    # Concrete
    ax.add_patch(Rectangle((px(0), py(0)), W_c, H_c,
        fc=CLR_CONCRETE, ec=CLR_CONCRETE_EDGE, lw=2.0, zorder=1))

    # BE zones
    lc_cv = lc_mm * sx
    for x0 in [0, lw_mm - lc_mm]:
        ax.add_patch(Rectangle((px(x0), py(0)), lc_cv, H_c,
            fc=CLR_BE_ZONE, ec=CLR_CONCRETE_EDGE,
            lw=0.8, alpha=0.50, hatch="///", zorder=2))

    for xv in [lc_mm, lw_mm - lc_mm]:
        ax.plot([px(xv), px(xv)], [py(0), py(bw_mm)],
                color=CLR_CONCRETE_EDGE, lw=1.2, zorder=3)

    # Cover dashed lines
    segs = [
        ([px(c), px(lw_mm-c)], [py(c), py(c)]),
        ([px(c), px(lw_mm-c)], [py(bw_mm-c), py(bw_mm-c)]),
        ([px(c), px(c)],       [py(c), py(bw_mm-c)]),
        ([px(lw_mm-c), px(lw_mm-c)], [py(c), py(bw_mm-c)]),
    ]
    for xs, ys in segs:
        ax.plot(xs, ys, color=CLR_DIM, lw=0.6, ls="--", zorder=4)

    # Confinement hoops (draw confined rectangle if ConfinedArea, else simple hoop)
    if cfg.confined is not None:
        ca = cfg.confined
        x_left_origin  = lc_mm / 2.0 - ca.w / 2.0
        x_right_origin = lw_mm - lc_mm / 2.0 - ca.w / 2.0
        y_origin       = bw_mm / 2.0 - ca.h / 2.0
        for x0 in [x_left_origin, x_right_origin]:
            ho_pad = c * 0.6
            ax.add_patch(Rectangle(
                (px(x0) + ho_pad * sx, py(y_origin) + ho_pad * sy),
                ca.w * sx - 2 * ho_pad * sx,
                ca.h * sy - 2 * ho_pad * sy,
                fc="none", ec=CLR_HATCH, lw=1.8, zorder=5))
    else:
        ho_x, ho_y = c * 0.85 * sx, c * 0.85 * sy
        for x0 in [0, lw_mm - lc_mm]:
            ax.add_patch(Rectangle(
                (px(x0) + ho_x, py(0) + ho_y),
                lc_cv - 2*ho_x, H_c - 2*ho_y,
                fc="none", ec=CLR_HATCH, lw=1.5, zorder=5))

    # Bars
    def r_cv(dia): return max(dia / 2 * sy, 2.2)

    for (bx, by, A) in be_bars:
        dia = np.sqrt(A / (np.pi / 4))
        ax.add_patch(Circle((px(bx), py(by)), r_cv(dia),
            fc=CLR_BE_BAR, ec="white", lw=0.6, zorder=7))

    for (bx, by, A) in web_bars:
        dia = np.sqrt(A / (np.pi / 4))
        ax.add_patch(Circle((px(bx), py(by)), r_cv(dia),
            fc=CLR_WEB_BAR, ec="white", lw=0.6, zorder=7))

    # Leader lines
    FS = 8.5
    top_be_left = [b for b in be_bars if b[0] < lc_mm and b[1] > bw_mm / 2]
    if top_be_left:
        bx, by, A = max(top_be_left, key=lambda b: b[1])
        dia = np.sqrt(A / (np.pi / 4))
        lbl = (f"Ø{cfg.confined.corner_dia}" if cfg.confined else
               f"Ø{cfg.boundary_dia}")
        ax.annotate(lbl, xy=(px(bx), py(by)),
                    xytext=(px(0) - 4, py(bw_mm) + 8),
                    fontsize=FS, color=CLR_BE_BAR, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=CLR_BE_BAR,
                                    lw=0.9, mutation_scale=9), zorder=9)

    if web_bars:
        wx, wy, _ = web_bars[len(web_bars) // 2]
        ax.annotate(f"Ø{cfg.web_dia}@{cfg.web_spacing}",
                    xy=(px(wx), py(wy)),
                    xytext=(px(lw_mm / 2), py(bw_mm) + 8),
                    fontsize=FS, color=CLR_WEB_BAR, fontweight="bold", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color=CLR_WEB_BAR,
                                    lw=0.9, mutation_scale=9), zorder=9)

    # Cover label
    ax.text(px(c / 2), py(bw_mm / 2), f"c={c:.0f}",
            ha="center", va="center", fontsize=6.5,
            color=CLR_DIM, rotation=90, style="italic")

    # Dimension lines
    gap = MB * 0.45
    def hdim(x1, x2, y_cv, lbl, fs=7.5):
        ax.annotate("", xy=(px(x2), y_cv), xytext=(px(x1), y_cv),
                    arrowprops=dict(arrowstyle="<->", color=CLR_DIM, lw=0.9))
        ax.text((px(x1)+px(x2))/2, y_cv - 4, lbl,
                ha="center", va="top", fontsize=fs, color=CLR_DIM)

    def vdim(x_cv, y1, y2, lbl, fs=7.5):
        ax.annotate("", xy=(x_cv, py(y2)), xytext=(x_cv, py(y1)),
                    arrowprops=dict(arrowstyle="<->", color=CLR_DIM, lw=0.9))
        ax.text(x_cv - 3, (py(y1)+py(y2))/2, lbl,
                ha="right", va="center", fontsize=fs, color=CLR_DIM, rotation=90)

    y1 = py(0) - gap * 0.9
    y2 = py(0) - gap * 1.9
    hdim(0,          lc_mm,     y1, f"lc={lc_mm:.0f} mm")
    hdim(lw_mm-lc_mm, lw_mm,   y1, f"lc={lc_mm:.0f} mm")
    hdim(0,           lw_mm,   y2, f"lw={lw_mm:.0f} mm")
    vdim(px(0) - 12, 0, bw_mm,     f"bw={bw_mm:.0f} mm")

    # Add ConfinedArea dimensions if present
    if cfg.confined is not None:
        ca = cfg.confined
        x0 = lc_mm / 2.0 - ca.w / 2.0
        hdim(x0, x0 + ca.w, py(0) - gap * 0.45, f"w={ca.w:.0f}", fs=6.5)
        vdim(px(x0) - 6, bw_mm/2 - ca.h/2, bw_mm/2 + ca.h/2,
             f"h={ca.h:.0f}", fs=6.5)

    # Steel area summary
    n_be  = len(be_bars)
    n_web = len(web_bars)
    A_be  = sum(b[2] for b in be_bars)
    A_web = sum(b[2] for b in web_bars)
    A_tot = A_be + A_web
    rho   = A_tot / (lw_mm * bw_mm) * 100

    if cfg.confined:
        be_desc = (f"{n_be//2}bars/side\n"
                   f"Ø{cfg.confined.corner_dia}+Ø{cfg.confined.along_dia}")
    else:
        be_desc = f"{n_be} × Ø{cfg.boundary_dia}"

    box = (
        f"As,BE  = {A_be/100:5.1f} cm²\n"
        f"         {be_desc}\n"
        f"As,web = {A_web/100:5.1f} cm²\n"
        f"         {n_web} × Ø{cfg.web_dia}\n"
        f"─────────────────\n"
        f"As,tot = {A_tot/100:5.1f} cm²\n"
        f"ρ      = {rho:.3f} %"
    )
    ax.text(px(lw_mm) + 6, py(bw_mm), box,
            ha="left", va="top", fontsize=8.5, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#aaa", lw=0.9))

    # Legend
    be_dia_lbl = (f"Ø{cfg.confined.corner_dia}" if cfg.confined else
                  f"Ø{cfg.boundary_dia}")
    handles = [
        mpatches.Patch(fc=CLR_CONCRETE, ec=CLR_CONCRETE_EDGE, label="Béton"),
        mpatches.Patch(fc=CLR_BE_ZONE,  ec=CLR_CONCRETE_EDGE, alpha=0.6,
                       hatch="///", label="Zone de rive"),
        mpatches.Patch(fc=CLR_BE_BAR,   label=f"Armatures BE  {be_dia_lbl}"),
        mpatches.Patch(fc=CLR_WEB_BAR,  label=f"Armatures âme Ø{cfg.web_dia}@{cfg.web_spacing}"),
        mlines.Line2D([], [], color=CLR_HATCH, lw=1.5, label="Cadre confinement"),
        mlines.Line2D([], [], color=CLR_DIM, lw=0.7, ls="--",
                      label=f"Enrobage {c:.0f} mm"),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=0.95,
              loc="upper left",
              bbox_to_anchor=(px(lw_mm) + 6, py(0) - 4),
              bbox_transform=ax.transData, borderpad=0.5, labelspacing=0.35)

    title_be = (f"w={cfg.confined.w:.0f}×h={cfg.confined.h:.0f}" if cfg.confined
                else f"{cfg.boundary_n}×Ø{cfg.boundary_dia}/face")
    ax.set_title(
        f"{cfg.name}  —  C30/FeE500   "
        f"lw={lw_mm:.0f}  bw={bw_mm:.0f}  lc={lc_mm:.0f}  c={c:.0f} mm  "
        f"[BE: {title_be}]",
        fontsize=9.5, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER RUN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_all(
    geom:       WallGeometry,
    mat:        Materials,
    combos:     list[LoadCombo],
    configs:    list[RebarConfig],
    storey_map: dict[int, str],
    nm_points:  int = 25,
    # Optional ductility params — pass to also compute §7.7.5
    ductility_params: Optional[dict] = None,
) -> RunResults:
    """Run all calculation phases and return a RunResults bundle.

    Parameters
    ----------
    ductility_params : dict | None
        If provided, must contain keys:
        ``T0``, ``T2``, ``R``, ``QF``, ``M_RD``
        and optionally ``M_ED``.
        Triggers §7.7.5 ductility calculations for the governing combo.
    """
    import pandas as pd

    phase1   = run_phase1(geom, mat, combos)
    envelope = build_envelope_table(geom, combos)

    concrete_mat, steel_mat = make_materials(mat)
    nm_curves = compute_nm_curves(geom, configs, concrete_mat, steel_mat, nm_points)
    storeys   = run_phase4(envelope, nm_curves, storey_map)

    base_cfg_name = storey_map.get(1) or storey_map[min(storey_map)]
    base_cfg      = next(c for c in configs if c.name == base_cfg_name)
    phase5        = run_phase5(geom, mat, combos, base_cfg)

    all_ok = all(r.ok for r in storeys)

    cols = {"Storey": [], "z [m]": [], "Menv max [kN·m]": []}
    for c in combos:
        cols[f"Menv {c.label} [kN·m]"] = []
    for row in envelope:
        cols["Storey"].append(row["storey"])
        cols["z [m]"].append(row["z_m"])
        cols["Menv max [kN·m]"].append(round(row["Menv_max"], 1))
        for c in combos:
            cols[f"Menv {c.label} [kN·m]"].append(round(row[f"Menv_{c.label}"], 1))
    env_df = pd.DataFrame(cols)

    nm_fig       = make_nm_figure(nm_curves, combos, storeys, geom)
    section_figs = {cfg.name: make_section_figure(cfg, geom) for cfg in configs}

    ductility     = None
    ductility_fig = None
    if ductility_params is not None:
        combo_gov = max(combos, key=lambda c: abs(c.N_ed))
        ductility = run_ductility(
            geom=geom, mat=mat, combo=combo_gov, cfg_base=base_cfg,
            T0=ductility_params["T0"],
            T2=ductility_params["T2"],
            R=ductility_params["R"],
            QF=ductility_params["QF"],
            M_RD=ductility_params["M_RD"],
            M_ED=ductility_params.get("M_ED"),
        )
        ductility_fig = make_ductility_figure(ductility, geom)

    return RunResults(
        phase1=phase1, envelope_df=env_df, storeys=storeys,
        phase5=phase5, all_ok=all_ok,
        nm_fig=nm_fig, section_figs=section_figs,
        ductility=ductility, ductility_fig=ductility_fig,
    )
