import numpy as np
import matplotlib.pyplot as plt
from engine import (
    WallGeometry, Materials, RebarConfig, ConfinedArea,
    make_materials, build_wall_section
)

# =============================================================================
# 1. SETUP PLOT STYLE (Larger Fonts & Cleaner Lines)
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "axes.titlesize": 15,       # Increased from 12
    "axes.labelsize": 14,       # Increased from 11
    "xtick.labelsize": 12,      # Increased tick labels
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150
})

# =============================================================================
# 2. DEFINE GEOMETRY, MATERIALS, AND REINFORCEMENT
# =============================================================================
geom = WallGeometry(lw=1.70, bw=0.30, hw=30.60, he=3.06, cover=30.0)
mat = Materials(fc28=30.0, fyk=500.0, Es=200_000.0, eps_su=0.05)

be_critique = ConfinedArea(
    w=450, h=300, corner_dia=20, along_2=2, along_3=1, along_dia=20
)
cfg = RebarConfig(
    name='C1 — Critique', confined=be_critique, web_dia=12, web_spacing=150
)

N_Ed = 12016.0  # kN (Compression)
M_Ed = 820.0   # kN.m

# =============================================================================
# 3. BUILD SECTION & COMPUTE N-M SURFACE
# =============================================================================
print("Building section and computing N-M diagram...")
concrete_mat, steel_mat = make_materials(mat)
sec = build_wall_section(geom, cfg, concrete_mat, steel_mat)
mi_result = sec.moment_interaction_diagram(theta=0, n_points=40, progress_bar=False)

N_curve = np.array([r.n for r in mi_result.results]) / 1000.0
M_curve = np.array([r.m_x for r in mi_result.results]) / 1000000.0

mask = N_curve >= 0
N_curve = N_curve[mask]
M_curve = M_curve[mask]

# =============================================================================
# 4. CALCULATE CAPACITIES
# =============================================================================
sort_idx = np.argsort(N_curve)
N_sorted = N_curve[sort_idx]
M_sorted = M_curve[sort_idx]

M_Rd_axial = np.interp(N_Ed, N_sorted, M_sorted)
eta_axial = M_Ed / M_Rd_axial

theta_curve = np.arctan2(N_curve, M_curve)
theta_demand = np.arctan2(N_Ed, M_Ed)

sort_idx_theta = np.argsort(theta_curve)
theta_sorted = theta_curve[sort_idx_theta]
M_theta_sorted = M_curve[sort_idx_theta]
N_theta_sorted = N_curve[sort_idx_theta]

M_Rd_radial = np.interp(theta_demand, theta_sorted, M_theta_sorted)
N_Rd_radial = np.interp(theta_demand, theta_sorted, N_theta_sorted)
eta_radial = M_Ed / M_Rd_radial

# =============================================================================
# 5. GENERATE THE PLOTS
# =============================================================================
# Increased figure size for more breathing room
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5))

# White background box style to prevent lines from crossing out text
text_mask = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9)

for ax in (ax1, ax2):
    ax.plot(M_curve, N_curve, color='black', lw=2.5, label='Capacity Surface ($M_{Rd}, N_{Rd}$)')
    ax.plot(M_Ed, N_Ed, marker='o', color='black', markersize=7, zorder=5)
    
    # Shifted demand label slightly so it doesn't overlap the marker
    ax.text(M_Ed + 40, N_Ed + 150, '$(M_{Ed}, N_{Ed})$', fontsize=13, zorder=10, bbox=text_mask)
    
    ax.set_xlim(0, max(M_curve) * 1.15)
    ax.set_ylim(0, max(N_curve) * 1.15)
    ax.set_xlabel('Bending Moment $M$ [kN·m]', weight='bold')
    ax.set_ylabel('Axial Compression $N$ [kN]', weight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
    ax.set_axisbelow(True)

# ---------------------------------------------------------
# Plot 1: Radial Method
# ---------------------------------------------------------
ax1.set_title('Constant Eccentricity (Radial) Method', pad=15)

ax1.plot([0, M_Rd_radial], [0, N_Rd_radial], 'k--', lw=1.5)
ax1.plot(M_Rd_radial, N_Rd_radial, marker='s', color='black', markersize=7, zorder=5)
ax1.text(M_Rd_radial + 40, N_Rd_radial - 250, '$(M_{Rd}, N_{Rd})$', fontsize=13, zorder=10, bbox=text_mask)

ax1.annotate('', xy=(M_Ed, N_Ed), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', lw=1.8, color='black'))
ax1.annotate('', xy=(M_Rd_radial, N_Rd_radial), xytext=(M_Ed, N_Ed),
             arrowprops=dict(arrowstyle='->', lw=1.8, color='gray', ls='-'))

# Placed text on top of the lines using the text_mask
ax1.text(M_Ed/2, N_Ed/2, r'$L_{Demand}$', fontsize=13, rotation=np.degrees(theta_demand), 
         ha='center', va='center', bbox=text_mask, zorder=10)
ax1.text((M_Ed + M_Rd_radial)/2, (N_Ed + N_Rd_radial)/2, r'$L_{Capacity}$', fontsize=13, color='gray', 
         rotation=np.degrees(theta_demand), ha='center', va='center', bbox=text_mask, zorder=10)

# Stats Box moved to Top Right
textstr1 = '\n'.join((
    r'$\eta = \frac{L_{Demand}}{L_{Capacity}} = %.2f$' % (eta_radial,),
    r'Yields artificially low ratio'
))
ax1.text(0.95, 0.95, textstr1, transform=ax1.transAxes, fontsize=13,
         verticalalignment='top', horizontalalignment='right', 
         bbox=dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='black'), zorder=10)

# ---------------------------------------------------------
# Plot 2: Constant Axial Load Method
# ---------------------------------------------------------
ax2.set_title('Constant Axial Load Method (Proposed)', pad=15)

ax2.axhline(N_Ed, color='black', lw=1.2, ls=':', zorder=4)
ax2.plot(M_Rd_axial, N_Ed, marker='s', color='black', markersize=7, zorder=5)
ax2.text(M_Rd_axial + 40, N_Ed + 150, '$(M_{Rd}, N_{Ed})$', fontsize=13, zorder=10, bbox=text_mask)

# Adjusted y-offsets for dimension lines so they are clear of the main data
y_offset_1 = N_Ed - max(N_curve) * 0.06
y_offset_2 = N_Ed - max(N_curve) * 0.12

ax2.annotate('', xy=(M_Ed, y_offset_1), xytext=(0, y_offset_1),
             arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
ax2.text(M_Ed/2, y_offset_1, r'$M_{Ed}$', ha='center', va='center', fontsize=13, bbox=text_mask, zorder=10)

ax2.annotate('', xy=(M_Rd_axial, y_offset_2), xytext=(0, y_offset_2),
             arrowprops=dict(arrowstyle='<->', lw=1.5, color='gray'))
ax2.text(M_Rd_axial/2, y_offset_2, r'$M_{Rd}$', ha='center', va='center', color='gray', fontsize=13, bbox=text_mask, zorder=10)

# Stats Box moved to Top Right
textstr2 = '\n'.join((
    r'$\eta = \frac{M_{Ed}}{M_{Rd}} = %.2f$' % (eta_axial,),
    r'Conservative & physically realistic'
))
ax2.text(0.95, 0.95, textstr2, transform=ax2.transAxes, fontsize=13,
         verticalalignment='top', horizontalalignment='right', 
         bbox=dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='black'), zorder=10)

# =============================================================================
# 6. FINALIZE & SAVE
# =============================================================================
plt.tight_layout(pad=3.0)
filename = "NM_Method_Comparison.pdf"
plt.savefig(filename, format='pdf', bbox_inches='tight')
print(f"Plot saved successfully as '{filename}'")
plt.show()