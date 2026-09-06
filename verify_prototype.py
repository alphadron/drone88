#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 verify_prototype.py — FacilityPath M2 프로토타입 자동 검증
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M2)
--------------------------------------------------------------------------------
 검증 항목 (개발계획서 §7 단위 검증):
   [V1] 합성 평면 메시 α = 0/30/45/60/75/90° → 분석 경사각 오차 ≤ 0.5°
   [V2] 짐벌 피치 φ = α − 90° 정확 일치 (오차 ≤ 0.5°)
   [V3] 헤딩 ψ = (aspect + 180°) mod 360 일치 (오차 ≤ 1°, 평지 제외)
   [V4] 표면 노이즈(±3 cm) 조건에서 목표 오차 ≤ 5° (개발계획서 정량 목표)
   [V5] 짐벌 가동범위 밖(상향 요구) 경고 발생 확인 (α=110° 오버행 모사 불가
        → 가동범위를 (-90,-40)으로 좁혀 α=90° 요구 φ=0° 위반 검출)
 산출물:
   verification_results.csv, prototype_demo_3d.png
================================================================================
"""

import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surface_analyzer import SurfaceAnalyzer, make_synthetic_slope
from tilt_solver import TiltSolver

TOL_EXACT = 0.5    # [V1][V2] 무노이즈 허용오차 [deg]
TOL_HEAD  = 1.0    # [V3] 헤딩 허용오차 [deg]
TOL_GOAL  = 5.0    # [V4] 개발계획서 정량 목표 [deg]


def sample_grid_on_surface(surf, n_u=6, n_v=4):
    """추출면의 u(등고선)×v(경사) 파라미터 공간에 검증용 격자점 생성."""
    n = surf.mean_normal
    up = np.array([0., 0., 1.])
    u = np.cross(up, n)
    if np.linalg.norm(u) < 1e-9:
        u = np.array([1., 0., 0.])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    us = np.linspace(-0.4, 0.4, n_u) * surf.extent_u_m
    vs = np.linspace(-0.4, 0.4, n_v) * surf.extent_v_m
    return np.array([surf.centroid + a * u + b * v for b in vs for a in us])


def ang_diff(a, b):
    """방위각 차이 [-180,180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def run_case(alpha, aspect, noise=0.0, tol_slope=TOL_EXACT, tol_pitch=TOL_EXACT):
    """단일 케이스: 메시 생성 → 분석 → 솔브 → 판정 딕셔너리 반환."""
    mesh = make_synthetic_slope(alpha, aspect, noise_m=noise)
    res = SurfaceAnalyzer(slope_min_deg=0.0, min_area_m2=1.0).analyze(mesh)
    surf = res.surfaces[0]

    pts = sample_grid_on_surface(surf)
    solver = TiltSolver(approach_dist_m=15.0, fit_radius_m=4.0,
                        smooth_window=3, max_pitch_rate=0.0)  # 검증은 rate 제한 끔
    sol = solver.solve(pts, mesh.vertices, surf.mean_normal)

    phi_exp = alpha - 90.0
    psi_exp = (aspect + 180.0) % 360.0
    pitches = np.array([w.gimbal_pitch_deg for w in sol.waypoints])
    heads = np.array([w.heading_deg for w in sol.waypoints])

    err_slope = abs(surf.mean_slope_deg - alpha)
    err_pitch = float(np.max(np.abs(pitches - phi_exp)))
    if alpha < 1e-6:                       # 평지: 헤딩 정의 불가 → NaN 확인
        head_ok = np.all(np.isnan(heads))
        err_head = 0.0
    else:
        err_head = float(np.max(np.abs(ang_diff(heads, psi_exp))))
        head_ok = err_head <= TOL_HEAD

    ok = (err_slope <= tol_slope) and (err_pitch <= tol_pitch) and head_ok
    return dict(alpha=alpha, aspect=aspect, noise=noise,
                slope_meas=round(surf.mean_slope_deg, 3),
                phi_expected=phi_exp,
                phi_meas_mean=round(float(pitches.mean()), 3),
                err_slope=round(err_slope, 3), err_pitch=round(err_pitch, 3),
                err_head=round(err_head, 3), result="PASS" if ok else "FAIL",
                _mesh=mesh, _surf=surf, _sol=sol)


def draw_demo(case, path="prototype_demo_3d.png"):
    """대표 케이스 3D 시각화: 메시 + 웨이포인트 + 광축 벡터."""
    mesh, sol = case["_mesh"], case["_sol"]
    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces,
                    color="#f5c518", alpha=0.45, edgecolor="#c9a30f",
                    linewidth=0.15)
    wp = np.array([w.position for w in sol.waypoints])
    sp = np.array([w.surface_point for w in sol.waypoints])
    ax.scatter(*wp.T, c="#0b7285", s=28, depthshade=False,
               label=f"Waypoints (d=15 m, n={len(wp)})")
    for w in sol.waypoints:                       # 광축(−n) 화살표
        v = -w.normal * 6.0
        ax.quiver(*w.position, *v, color="#c92a2a",
                  arrow_length_ratio=0.18, linewidth=1.1)
    ax.plot(*np.vstack([wp, sp[::-1]])[:0].T)     # (자리표시)
    a, asp = case["alpha"], case["aspect"]
    ax.set_title(f"FacilityPath M2 Prototype — Slope α={a}°, aspect={asp}°  "
                 f"→  Gimbal φ={a-90:.0f}°  (φ = α − 90°)",
                 fontsize=11, color="#12303c", weight="bold")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((1, 1, 0.55))
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  → 3D 데모 저장: {path}")


def main():
    print("=" * 84)
    print("  FacilityPath M2 프로토타입 검증 — surface_analyzer + tilt_solver")
    print("  핵심 공식: 짐벌 피치 φ = α − 90°  /  헤딩 ψ = (aspect + 180°) mod 360")
    print("=" * 84)

    rows, fails = [], 0

    # ── [V1~V3] 무노이즈 정확 일치 ──
    print("\n[V1~V3] 합성 평면 메시 — 무노이즈, 허용오차 0.5°(헤딩 1°)")
    print(f"  {'α[°]':>6} {'aspect':>7} | {'측정α':>8} {'기대φ':>7} {'측정φ':>8} "
          f"| {'Δα':>6} {'Δφ':>6} {'Δψ':>6} | 판정")
    cases = [(0, 180), (30, 90), (45, 180), (60, 180), (60, 315), (75, 45), (90, 180)]
    for alpha, aspect in cases:
        c = run_case(alpha, aspect)
        rows.append(c)
        fails += c["result"] == "FAIL"
        print(f"  {c['alpha']:>6} {c['aspect']:>7} | {c['slope_meas']:>8.3f} "
              f"{c['phi_expected']:>7.1f} {c['phi_meas_mean']:>8.3f} "
              f"| {c['err_slope']:>6.3f} {c['err_pitch']:>6.3f} "
              f"{c['err_head']:>6.3f} | {c['result']}")

    # ── [V4] 노이즈 강건성 (±3 cm) — 목표 ≤ 5° ──
    print(f"\n[V4] 표면 노이즈 ±3 cm — 개발계획서 정량 목표(법선 정렬 오차 ≤ {TOL_GOAL}°)")
    for alpha, aspect in [(45, 180), (70, 240)]:
        c = run_case(alpha, aspect, noise=0.03,
                     tol_slope=TOL_GOAL, tol_pitch=TOL_GOAL)
        rows.append(c)
        fails += c["result"] == "FAIL"
        print(f"  α={alpha}° aspect={aspect}° noise=3cm → "
              f"Δφ(max)={c['err_pitch']:.3f}°  {c['result']}")

    # ── [V5] 짐벌 가동범위 경고 ──
    print("\n[V5] 짐벌 가동범위 경고 — 범위 (-90,-40)°에서 수직벽(α=90°, 요구 φ=0°)")
    mesh = make_synthetic_slope(90, 180)
    surf = SurfaceAnalyzer(slope_min_deg=0, min_area_m2=1).analyze(mesh).surfaces[0]
    solver = TiltSolver(gimbal_range=(-90.0, -40.0), max_pitch_rate=0.0)
    solver.solve(sample_grid_on_surface(surf), mesh.vertices, surf.mean_normal)
    v5_ok = len(solver.warnings) > 0
    fails += not v5_ok
    print(f"  경고 {len(solver.warnings)}건 발생 → {'PASS' if v5_ok else 'FAIL'} "
          f"(예: {solver.warnings[0] if solver.warnings else '-'})")

    # ── CSV 저장 ──
    with open("verification_results.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["α[deg]", "aspect[deg]", "noise[m]", "측정 경사각[deg]",
                    "기대 φ[deg]", "측정 φ 평균[deg]", "Δ경사각", "Δ피치(max)",
                    "Δ헤딩(max)", "판정"])
        for c in rows:
            w.writerow([c["alpha"], c["aspect"], c["noise"], c["slope_meas"],
                        c["phi_expected"], c["phi_meas_mean"], c["err_slope"],
                        c["err_pitch"], c["err_head"], c["result"]])
    print("\n  → CSV 저장: verification_results.csv")

    # ── 대표 케이스(α=60°) 3D 데모 ──
    demo = next(c for c in rows if c["alpha"] == 60 and c["aspect"] == 180)
    draw_demo(demo)

    total = len(rows) + 1
    print("\n" + "=" * 84)
    print(f"  검증 결과: {total - fails}/{total} PASS"
          + ("  ✓ 전 항목 통과" if fails == 0 else f"  ✗ {fails}건 실패"))
    print("=" * 84)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
