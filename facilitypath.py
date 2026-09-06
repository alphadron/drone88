#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 facilitypath.py — FacilityPath v1.1 통합 실행 진입점
 시설물 안전점검 드론 자동비행경로 생성 (급경사지 / 교량 / 댐)
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · DID-DP-2026-0829 부속 (2026-09-05)
--------------------------------------------------------------------------------
 처리 순서 (모든 모드 공통 — 모듈 파일명이 단계와 1:1 대응):

   [1] 기준면 입력      io_model.py        OBJ/PLY 메시 또는 DSM → 로컬 ENU 메시
   [2] 대상면 분석      surface_analyzer   경사각 범위로 촬영면 추출, α·주향·범위
   [3] 촬영 파라미터    camera.py          GSD ↔ 접근거리, 트리거/라인 간격
   [4] 경로 생성        ┌ generate 모드: planner_slope.py  (형상 기반 신규 설계)
                        └ adapt    모드: io_flightpath.py → path_adapter.py
                                          (기존 그리드/파사드 경로를 법선 정렬로 재배치)
       짐벌 산출        tilt_solver.py     φ = α − 90°, 헤딩 = 면을 향함
   [5] 확정             planner_base.finalize   최소이격 검사 → 소티 분할 → 통계
   [6] 출력             kml_writer.py      Earth 검토 KML + 소티별 DJI WPML(.kmz)
                                          + 웨이포인트 CSV + 요약 JSON

 사용 예:
   # 3D 모델(투영좌표 EPSG:5186) → 급경사지 경로 신규 생성
   python facilitypath.py generate slope_model.obj --epsg 5186 --gsd 0.5 --out run1

   # DSM(ASC, EPSG:5186) → 신규 생성 (격자 2칸 간격 데시메이션)
   python facilitypath.py generate site_dsm.asc --epsg 5186 --dsm-step 2 --out run2

   # 기존 UgCS 그리드 KML → 사면 형상에 맞게 법선 정렬 재배치
   python facilitypath.py adapt ugcs_grid.kml --model slope_model.obj --epsg 5186 \\
          --gsd 0.5 --alt-mode absolute --out run3

   # 로컬 좌표 OBJ는 기준점 지정 필요
   python facilitypath.py generate local.obj --anchor 36.2243694,127.2765611,121 --out run4
================================================================================
"""

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from kml_writer import GeoAnchor, export_mission
from io_model import load_reference
from io_flightpath import load_flightpath, load_boundary_polygon
from surface_analyzer import SurfaceAnalyzer
from camera import get_camera
from planner_base import PlanConfig, clip_to_boundary
from planner_slope import SlopePlanner
from path_adapter import PathAdapter, AdaptConfig


# ──────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def banner(step, title):
    print(f"\n[{step}] {title}")
    print("    " + "-" * 70)


def parse_anchor(s):
    if not s:
        return None
    lat, lon, h = [float(x) for x in s.split(",")]
    return GeoAnchor(lat=lat, lon=lon, h=h)


def make_plan_config(a) -> PlanConfig:
    return PlanConfig(gsd_m=a.gsd / 100, fwd_overlap=a.fwd, side_overlap=a.side,
                      speed_ms=a.speed, min_clearance_m=a.clearance,
                      safe_endurance_min=a.endurance,
                      max_pitch_rate=a.pitch_rate)


def write_csv(res, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["WP", "소티", "라인", "E[m]", "N[m]", "U[m]",
                    "짐벌피치[deg]", "헤딩[deg]", "국부경사각[deg]"])
        for i, wp in enumerate(res.waypoints):
            w.writerow([i, int(res.sortie_index[i]), int(res.line_index[i]),
                        *np.round(wp.position, 2), round(wp.gimbal_pitch_deg, 2),
                        round(wp.heading_deg, 2), round(wp.slope_deg, 2)])


def summarize(res, ref, mode, surf=None):
    s = dict(mode=mode, source=ref.source, source_kind=ref.kind,
             anchor=dict(lat=ref.anchor.lat, lon=ref.anchor.lon, h=ref.anchor.h),
             n_waypoints=len(res.waypoints), n_lines=res.n_lines,
             n_sorties=res.n_sorties, approach_dist_m=round(res.approach_dist_m, 2),
             trigger_spacing_m=round(res.trigger_spacing_m, 2),
             line_spacing_m=round(res.line_spacing_m, 2),
             path_len_m=round(res.path_len_m, 1),
             flight_time_min=round(res.flight_time_min, 1),
             data_gb=round(res.data_gb, 1),
             min_clearance_m=round(getattr(res, "min_clearance_measured", 0), 2),
             pitch_range_deg=[round(min(w.gimbal_pitch_deg for w in res.waypoints), 1),
                              round(max(w.gimbal_pitch_deg for w in res.waypoints), 1)],
             warnings=res.warnings)
    if surf is not None:
        s["surface"] = dict(slope_deg=round(surf.mean_slope_deg, 2),
                            aspect_deg=round(surf.mean_aspect_deg, 2),
                            area_m2=round(surf.area_m2, 1),
                            extent_m=[round(surf.extent_u_m, 1), round(surf.extent_v_m, 1)])
    return s


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인
# ──────────────────────────────────────────────────────────────────────────────
def run(a):
    os.makedirs(a.out, exist_ok=True)
    prefix = os.path.join(a.out, "mission")

    # [1] 기준면 입력 ------------------------------------------------------------
    banner(1, "기준면 입력 (io_model)")
    ref_path = a.model if a.mode == "adapt" else a.input
    ref = load_reference(ref_path, epsg=a.epsg, anchor=parse_anchor(a.anchor),
                         dsm_step=a.dsm_step)
    print(f"    {ref.kind.upper():4s} {ref.source} → 정점 {len(ref.mesh.vertices):,} / "
          f"면 {len(ref.mesh.faces):,}")
    print(f"    ENU 원점: {ref.anchor.lat:.7f}N {ref.anchor.lon:.7f}E h={ref.anchor.h:.1f} m")

    boundary_en = None
    if a.boundary:
        boundary_en = load_boundary_polygon(a.boundary, ref.anchor)
        print(f"    촬영 경계: {a.boundary} ({len(boundary_en)}점, ENU 원점 기준)")

    # [2] 대상면 분석 --------------------------------------------------------------
    banner(2, "대상면 분석 (surface_analyzer)")
    analyzer = SurfaceAnalyzer(slope_min_deg=a.slope_min, slope_max_deg=a.slope_max,
                               min_area_m2=a.min_area)
    res_an = analyzer.analyze(ref.mesh)
    if not res_an.surfaces:
        sys.exit(f"    ✗ 경사각 {a.slope_min}~{a.slope_max}° 범위의 대상면이 없습니다. "
                 f"--slope-min/--slope-max 조정 필요")
    for i, s in enumerate(res_an.surfaces[:5]):
        print(f"    #{i} {s}")
    surf = res_an.surfaces[a.surface_index]
    print(f"    → 선택 대상면 #{a.surface_index}")

    # [3] 촬영 파라미터 ------------------------------------------------------------
    banner(3, "촬영 파라미터 (camera)")
    cam = get_camera(a.camera)
    cfg = make_plan_config(a)
    d = cam.dist_for_gsd(cfg.gsd_m)
    trig, line = cam.spacing(cfg.gsd_m, cfg.fwd_overlap, cfg.side_overlap)
    print(f"    {cam.name} | GSD {a.gsd} cm → 접근거리 {d:.2f} m | "
          f"트리거 {trig:.2f} m / 라인 {line:.2f} m | 중복 {a.fwd*100:.0f}/{a.side*100:.0f}%")

    # [4] 경로 생성 ----------------------------------------------------------------
    planner = SlopePlanner(cam, cfg)
    if a.mode == "generate":
        banner(4, f"경로 신규 생성 (planner_slope, 시설물={a.facility})")
        if a.facility != "slope":
            print(f"    ⚠ {a.facility} 전략은 M3 후속 구현 예정 — slope 전략으로 대체 실행")
        res = planner.plan(surf, ref.mesh.vertices, boundary_en=boundary_en)
    else:
        banner(4, "기존 경로 적응 (io_flightpath → path_adapter)")
        ip = load_flightpath(a.input, ref.anchor, alt_mode=a.alt_mode,
                             takeoff_z=a.takeoff_z)
        print(f"    입력 경로: {ip.fmt.upper()} {ip.source} ({ip.note})")
        path_center = ip.enu[:, :2].mean(axis=0)
        model_center = ref.mesh.vertices[:, :2].mean(axis=0)
        center_dist = float(np.hypot(*(path_center - model_center)))
        if center_dist > 500.0:
            print(f"    ⚠ 기준면-입력 경로 중심 거리 {center_dist:.0f} m — 기준면과 "
                  f"입력 경로가 같은 현장을 가리키는지 --anchor/--epsg 값을 확인하십시오")
        adapter = PathAdapter(cam, cfg, AdaptConfig(
            approach_dist_m=None if a.approach is None else a.approach,
            keep_distance=a.keep_distance, max_snap_m=a.max_snap))
        res = adapter.adapt(ip.enu, ref.mesh)
        if boundary_en is not None:
            res = clip_to_boundary(res, boundary_en)
        res.line_spacing_m = line
        if res.waypoints:
            res = planner.finalize(res, ref.mesh.vertices)      # [5] 공용 확정

    # [5] 확정 결과 ----------------------------------------------------------------
    banner(5, "확정 (finalize: 최소이격 → 소티 → 통계)")
    if not res.waypoints:
        sys.exit("    ✗ 생성된 웨이포인트가 없습니다")
    phis = [w.gimbal_pitch_deg for w in res.waypoints]
    print(f"    웨이포인트 {len(res.waypoints):,} / 라인 {res.n_lines} / 소티 {res.n_sorties}")
    print(f"    경로 {res.path_len_m/1000:.2f} km / 비행 {res.flight_time_min:.1f} 분 / "
          f"데이터 {res.data_gb:.1f} GB")
    print(f"    짐벌 피치 {min(phis):.1f}° ~ {max(phis):.1f}° | "
          f"표면 최소이격 {getattr(res,'min_clearance_measured',0):.2f} m")
    for w in res.warnings[:8]:
        print(f"    ⚠ {w}")

    # [6] 출력 ---------------------------------------------------------------------
    banner(6, "출력 (kml_writer)")
    mission_name = a.mission_name or (
        f"FacilityPath {a.mode} — {os.path.basename(a.input)}")
    files, rt = export_mission(res.waypoints, res.sortie_index, ref.anchor,
                               out_prefix=prefix, speed_ms=cfg.speed_ms,
                               takeoff_z_m=a.takeoff_z, surface_mesh=ref.mesh,
                               wp_label_step=a.wp_label_step,
                               altitude_mode=a.kml_altmode,
                               mission_name=mission_name, wp_extrude=a.wp_extrude,
                               show_anchor_marker=a.show_anchor_marker,
                               boundary_en=boundary_en)
    write_csv(res, prefix + "_waypoints.csv"); files.append(prefix + "_waypoints.csv")
    summ = summarize(res, ref, a.mode, surf)
    summ["enu_roundtrip_err_m"] = rt
    with open(prefix + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    files.append(prefix + "_summary.json")
    for fpath in files:
        print(f"    → {fpath}")
    print(f"    ENU↔WGS84 왕복오차 {rt*1000:.4f} mm")
    print("\n완료.")
    return summ


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="FacilityPath — 시설물 드론 자동비행경로 생성")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        g = sp.add_argument_group("기준면/좌표")
        g.add_argument("--epsg", type=int, help="입력 투영좌표계 (예: 5186 중부, 5187 동부)")
        g.add_argument("--anchor", help="로컬 좌표용 기준점 'lat,lon,h'")
        g.add_argument("--dsm-step", type=int, default=1, help="DSM 데시메이션 간격")
        g.add_argument("--slope-min", type=float, default=25.0)
        g.add_argument("--slope-max", type=float, default=90.0)
        g.add_argument("--min-area", type=float, default=10.0)
        g.add_argument("--surface-index", type=int, default=0, help="대상면 선택(면적순)")
        g.add_argument("--boundary",
                       help="촬영 경계 KML(Google Earth Pro '다각형 추가'로 실제 "
                            "법면 범위를 그려 내보낸 파일) — 지정 시 다각형 밖 "
                            "웨이포인트를 제외한다")
        c = sp.add_argument_group("촬영")
        c.add_argument("--camera", default="P1_50mm")
        c.add_argument("--gsd", type=float, default=0.5, help="목표 GSD [cm]")
        c.add_argument("--fwd", type=float, default=0.80)
        c.add_argument("--side", type=float, default=0.70)
        c.add_argument("--speed", type=float, default=2.5)
        c.add_argument("--pitch-rate", type=float, default=10.0)
        s = sp.add_argument_group("안전/운용")
        s.add_argument("--clearance", type=float, default=5.0, help="표면 최소 이격 [m]")
        s.add_argument("--endurance", type=float, default=25.0, help="소티 한계 [분]")
        s.add_argument("--takeoff-z", type=float, default=0.0, help="이륙점 ENU z [m]")
        v = sp.add_argument_group("Earth 검토 KML")
        v.add_argument("--wp-label-step", type=int, default=5,
                       help="웨이포인트 이름 라벨 표시 간격(1=전 웨이포인트 표시)")
        v.add_argument("--kml-altmode", choices=["absolute", "relativeToGround"],
                       default="absolute",
                       help="review KML 고도 기준. absolute=기준점 타원체고+z"
                            "(기준점 고도가 부정확하면 지형에 파묻혀 안 보일 수 있음), "
                            "relativeToGround=실제 지표 기준(항상 지표 위에 표시)")
        v.add_argument("--mission-name", default=None,
                       help="review KML Document 이름(미지정 시 입력 파일명+모드로 "
                            "자동 생성) — 여러 결과물을 Earth Pro에 동시에 열어도 "
                            "Places 패널에서 구분되도록 실행마다 다르게 남는다")
        v.add_argument("--wp-extrude", action=argparse.BooleanOptionalAction, default=True,
                       help="웨이포인트마다 지면까지 수직 안내선 표시(기본 켜짐). "
                            "기준점 고도가 실제 지형과 크게 다르면 안내선이 비정상적으로 "
                            "길게 그려질 수 있으므로 --no-wp-extrude로 끌 수 있다")
        v.add_argument("--show-anchor-marker", action=argparse.BooleanOptionalAction,
                       default=True,
                       help="ENU 원점(기준점)을 지면 고정 핀으로 표시(기본 켜짐) — "
                            "기준점이 실제 시설물 위치와 다른지 화면에서 바로 확인용")
        sp.add_argument("--out", default="output", help="출력 폴더")

    g = sub.add_parser("generate", help="3D 모델/DSM → 경로 신규 생성")
    g.add_argument("input", help="OBJ/PLY/STL 메시 또는 DSM(.tif/.asc/.xyz)")
    g.add_argument("--facility", choices=["slope", "bridge", "dam"], default="slope")
    common(g)

    ad = sub.add_parser("adapt", help="기존 경로(KML/KMZ/CSV) → 시설물 형상 적응")
    ad.add_argument("input", help="기존 경로 파일 (.kml/.kmz/.csv)")
    ad.add_argument("--model", required=True, help="기준면 메시 또는 DSM")
    ad.add_argument("--alt-mode", choices=["absolute", "relative"], default="relative")
    ad.add_argument("--approach", type=float, default=None,
                    help="접근거리 [m] (미지정 시 GSD로 산출)")
    ad.add_argument("--keep-distance", action="store_true",
                    help="원래 이격거리 유지(법선만 정렬)")
    ad.add_argument("--max-snap", type=float, default=80.0,
                    help="기준면에서 이보다 먼 점은 제외 [m]")
    common(ad)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
