#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 verify_planner_slope.py — FacilityPath M3 급경사지 플래너 자동 검증
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M3)
--------------------------------------------------------------------------------
 검증 항목:
   [P1] 라인 간격 ≤ 이론 라인 간격(횡중복 반영), 균등성 (변동 ≤ 1%)
   [P2] 트리거 간격 ≤ 이론 트리거 간격(종중복 반영)
   [P3] 전 웨이포인트 짐벌 피치 = α − 90° (무노이즈, 오차 ≤ 0.5°)
   [P4] 서펜타인 정렬: 인접 라인 진행방향 반전 확인
   [P5] 안전 이격: 최소 표면거리 ≈ 접근거리, ≥ 안전버퍼
   [P6] 커버리지: 라인 적층 폭 ≥ 면 경사방향 범위 (여유 포함)
   [P7] 소티 분할: 안전한계 축소 시 다소티 분할 + 라인 경계 절단 확인
 산출물:
   slope_path_3d.png, slope_waypoints.csv, slope_plan_summary(콘솔)
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

from camera import get_camera
from surface_analyzer import SurfaceAnalyzer, make_synthetic_slope
from planner_base import PlanConfig
from planner_slope import SlopePlanner

ALPHA, ASPECT = 60.0, 180.0          # 검증 기준 사면: 60° 남향
WIDTH, LENGTH = 120.0, 40.0          # 폭 120 m × 사면길이 40 m (절토사면 축소 모형)


def build_plan(gsd_cm=0.5, endurance_min=25.0):
    mesh = make_synthetic_slope(ALPHA, ASPECT, width_m=WIDTH, length_m=LENGTH,
                                grid=40)
    surf = SurfaceAnalyzer(slope_min_deg=10, min_area_m2=10).analyze(mesh).surfaces[0]
    cam = get_camera("P1_50mm")
    cfg = PlanConfig(gsd_m=gsd_cm / 100, fwd_overlap=0.80, side_overlap=0.70,
                     speed_ms=2.5, min_clearance_m=5.0,
                     safe_endurance_min=endurance_min, max_pitch_rate=0.0)
    res = SlopePlanner(cam, cfg).plan(surf, mesh.vertices)
    return mesh, surf, cam, cfg, res


def main():
    print("=" * 86)
    print("  FacilityPath M3 검증 — planner_slope (급경사지 등고선형 서펜타인)")
    print(f"  기준 사면: α={ALPHA}°, aspect={ASPECT}°, {WIDTH}×{LENGTH} m / "
          f"P1 50mm, GSD 0.5 cm, 중복 80/70%")
    print("=" * 86)

    mesh, surf, cam, cfg, res = build_plan()
    pos = res.positions()
    checks = []

    # ── 산출 요약 ──
    print(f"\n  접근거리 d          : {res.approach_dist_m:8.2f} m")
    print(f"  트리거/라인 간격     : {res.trigger_spacing_m:8.2f} / "
          f"{res.line_spacing_m:.2f} m")
    print(f"  라인 수 / 웨이포인트 : {res.n_lines:8d} / {len(res.waypoints):,}")
    print(f"  경로 길이 / 비행시간 : {res.path_len_m/1000:8.2f} km / "
          f"{res.flight_time_min:.1f} 분")
    print(f"  소티 / 데이터량      : {res.n_sorties:8d} / {res.data_gb:.1f} GB")
    print(f"  표면 최소이격(실측)  : {res.min_clearance_measured:8.2f} m")

    # ── [P1] 라인 간격 ──
    u, v = SlopePlanner.surface_axes(surf)
    line_v = [float(np.mean((pos[res.line_index == i] - surf.centroid) @ v))
              for i in range(res.n_lines)]
    gaps = np.diff(sorted(line_v))
    p1 = gaps.max() <= res.line_spacing_m * 1.001 and \
         (gaps.max() - gaps.min()) / gaps.mean() <= 0.01
    checks.append(("P1 라인 간격·균등성", p1,
                   f"max {gaps.max():.2f} ≤ {res.line_spacing_m:.2f} m, "
                   f"변동 {(gaps.max()-gaps.min())/gaps.mean()*100:.2f}%"))

    # ── [P2] 트리거 간격 ──
    d0 = np.linalg.norm(np.diff(pos[res.line_index == 0], axis=0), axis=1)
    p2 = d0.max() <= res.trigger_spacing_m * 1.001
    checks.append(("P2 트리거 간격", p2,
                   f"max {d0.max():.2f} ≤ {res.trigger_spacing_m:.2f} m"))

    # ── [P3] 짐벌 피치 = α − 90° ──
    phi = np.array([w.gimbal_pitch_deg for w in res.waypoints])
    err = np.abs(phi - (ALPHA - 90.0)).max()
    checks.append(("P3 짐벌 피치 φ=α−90°", err <= 0.5,
                   f"φ 평균 {phi.mean():.3f}° (기대 {ALPHA-90:.0f}°), "
                   f"최대오차 {err:.3f}°"))

    # ── [P4] 서펜타인 반전 ──
    dirs = []
    for i in range(res.n_lines):
        lp = pos[res.line_index == i]
        dirs.append((lp[-1] - lp[0]) / np.linalg.norm(lp[-1] - lp[0]))
    dots = [float(np.dot(dirs[i], dirs[i + 1])) for i in range(len(dirs) - 1)]
    p4 = all(dt < -0.99 for dt in dots)
    checks.append(("P4 서펜타인 방향 반전", p4,
                   f"인접 라인 방향내적 max {max(dots):.3f} (< −0.99)"))

    # ── [P5] 안전 이격 ──
    p5 = (abs(res.min_clearance_measured - res.approach_dist_m) <= 1.0
          and res.min_clearance_measured >= cfg.min_clearance_m
          and not any("이격 위반" in w for w in res.warnings))
    checks.append(("P5 표면 최소이격", p5,
                   f"실측 {res.min_clearance_measured:.2f} m ≈ d "
                   f"{res.approach_dist_m:.2f} m, 버퍼 {cfg.min_clearance_m} m ✓"))

    # ── [P6] 커버리지 ──
    span_v = max(line_v) - min(line_v)
    p6 = span_v >= surf.extent_v_m
    checks.append(("P6 경사방향 커버리지", p6,
                   f"적층 폭 {span_v:.1f} m ≥ 면 범위 {surf.extent_v_m:.1f} m"))

    # ── [P7] 소티 분할 (한계 3분으로 축소) ──
    *_, res_s = build_plan(endurance_min=3.0)
    cut_ok = all(
        res_s.line_index[i] != res_s.line_index[i - 1]
        for i in range(1, len(res_s.waypoints))
        if res_s.sortie_index[i] != res_s.sortie_index[i - 1])
    p7 = res_s.n_sorties >= 2 and cut_ok
    checks.append(("P7 소티 분할·라인경계 절단", p7,
                   f"한계 3분 → {res_s.n_sorties}소티, 절단점 전부 라인 경계 ✓"))

    # ── 판정 출력 ──
    print("\n  ── 검증 판정 ──")
    fails = 0
    for name, ok, detail in checks:
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24s} {detail}")

    # ── CSV: 웨이포인트 목록 ──
    with open("slope_waypoints.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["WP", "소티", "라인", "E[m]", "N[m]", "U[m]",
                    "짐벌피치[deg]", "헤딩[deg]", "국부경사각[deg]"])
        for i, wp in enumerate(res.waypoints):
            w.writerow([i, res.sortie_index[i], res.line_index[i],
                        *np.round(wp.position, 2),
                        round(wp.gimbal_pitch_deg, 2), round(wp.heading_deg, 2),
                        round(wp.slope_deg, 2)])
    print("\n  → CSV 저장: slope_waypoints.csv")

    # ── 3D 시각화 ──
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color="#f5c518",
                    alpha=0.40, edgecolor="#c9a30f", linewidth=0.1)
    cmap = plt.get_cmap("winter")
    for s in range(res.n_sorties):
        sel = res.sortie_index == s
        ax.plot(*pos[sel].T, "-", color=cmap(s / max(1, res.n_sorties - 1)),
                linewidth=1.6, label=f"Sortie {s+1}")
    step = max(1, len(res.waypoints) // 60)          # 화살표 과밀 방지
    for wp in res.waypoints[::step]:
        ax.quiver(*wp.position, *(-wp.normal * 5), color="#c92a2a",
                  arrow_length_ratio=0.25, linewidth=0.9)
    ax.scatter(*pos[0], c="#12303c", s=60, marker="^", label="Start")
    ax.set_title(f"FacilityPath M3 — SlopePlanner  (α={ALPHA:.0f}°, "
                 f"φ={ALPHA-90:.0f}°, d={res.approach_dist_m:.1f} m, "
                 f"{res.n_lines} lines / {len(res.waypoints)} WPs)",
                 fontsize=11, color="#12303c", weight="bold")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=24, azim=-60)
    ax.set_box_aspect((2.2, 1, 0.8))
    plt.tight_layout()
    plt.savefig("slope_path_3d.png", dpi=140)
    print("  → 3D 시각화 저장: slope_path_3d.png")

    total = len(checks)
    print("\n" + "=" * 86)
    print(f"  검증 결과: {total - fails}/{total} PASS"
          + ("  ✓ 전 항목 통과" if fails == 0 else f"  ✗ {fails}건 실패"))
    print("=" * 86)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
