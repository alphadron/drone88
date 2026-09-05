#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 planner_base.py — FacilityPath v1.0 / M3 전략 계층
 시설물별 경로 생성 전략의 공통 추상 클래스 (개발계획서 §3.1, §4.3)
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M3)
--------------------------------------------------------------------------------
 공통 파이프라인:
   대상면 추출(SurfaceAnalyzer) → 2D 파라미터화(u,v) → 격자 설계(중복도)
   → 법선 오프셋·짐벌각 산출(TiltSolver) → 서펜타인 정렬 → 안전검사
   → 소티 분할 → PathResult
 시설물별 전략(급경사지/교량/댐)은 design_lines() 만 재정의한다.
================================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from camera import CameraSpec
from surface_analyzer import SurfaceInfo
from tilt_solver import TiltSolver, Waypoint


# ──────────────────────────────────────────────────────────────────────────────
# 파라미터 / 결과
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PlanConfig:
    gsd_m: float = 0.005            # 목표 GSD [m] (접근거리와 양자택일 — GSD 우선)
    fwd_overlap: float = 0.80       # 종중복도 (진행 방향)
    side_overlap: float = 0.70      # 횡중복도 (라인 간격)
    speed_ms: float = 2.5           # 비행속도 [m/s]
    margin_frac: float = 0.05       # 면 가장자리 커버 여유 (extent 비율)
    min_clearance_m: float = 5.0    # 표면 최소 이격 안전버퍼 [m]
    safe_endurance_min: float = 25.0  # 소티 안전한계 [분]
    wp_overhead_s: float = 1.0      # 웨이포인트당 감가속/촬영 오버헤드 [s]
    long_side_cross: bool = True    # 장변을 라인 직교 방향으로 배치
    fit_radius_m: float = 4.0       # 국부 법선 추정 반경 [m]
    smooth_window: int = 3
    max_pitch_rate: float = 10.0
    gimbal_range: tuple = (-120.0, 30.0)


@dataclass
class PathResult:
    waypoints: list = field(default_factory=list)   # Waypoint (서펜타인 순서)
    line_index: np.ndarray = None                   # 웨이포인트별 라인 번호
    sortie_index: np.ndarray = None                 # 웨이포인트별 소티 번호
    n_lines: int = 0
    approach_dist_m: float = 0.0
    trigger_spacing_m: float = 0.0
    line_spacing_m: float = 0.0
    path_len_m: float = 0.0
    flight_time_min: float = 0.0
    n_sorties: int = 1
    data_gb: float = 0.0
    warnings: list = field(default_factory=list)

    def positions(self) -> np.ndarray:
        return np.array([w.position for w in self.waypoints])


# ──────────────────────────────────────────────────────────────────────────────
# 추상 플래너
# ──────────────────────────────────────────────────────────────────────────────
class PathPlanner(ABC):
    """시설물별 경로 전략의 공통 골격. 서브클래스는 design_lines()만 구현."""

    def __init__(self, camera: CameraSpec, config: PlanConfig = None):
        self.cam = camera
        self.cfg = config or PlanConfig()

    # ── 면의 2D 파라미터 축 (u: 등고선/수평, v: 오르막) ──
    @staticmethod
    def surface_axes(surf: SurfaceInfo) -> tuple:
        n = surf.mean_normal
        up = np.array([0.0, 0.0, 1.0])
        u = np.cross(up, n)
        if np.linalg.norm(u) < 1e-9:            # 평지: 임의 수평축
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        return u, v

    # ── 시설물별 라인 설계 (서브클래스 구현) ──
    @abstractmethod
    def design_lines(self, surf: SurfaceInfo, trigger_m: float,
                     line_m: float) -> list:
        """
        면 파라미터 공간에서 촬영 라인 목록을 설계한다.
        반환: [np.ndarray (Ni,3) 면상점 시퀀스, ...]  — 라인 순서대로
        """

    # ── 공통 실행 ──
    def plan(self, surf: SurfaceInfo, mesh_vertices: np.ndarray) -> PathResult:
        cfg, cam = self.cfg, self.cam
        d = cam.dist_for_gsd(cfg.gsd_m)
        trig, line = cam.spacing(cfg.gsd_m, cfg.fwd_overlap, cfg.side_overlap,
                                 cfg.long_side_cross)

        res = PathResult(approach_dist_m=d, trigger_spacing_m=trig,
                         line_spacing_m=line)

        # 접근거리 < 안전버퍼면 물리적으로 불가 — 즉시 경고
        if d < cfg.min_clearance_m:
            res.warnings.append(
                f"[안전] 접근거리 {d:.1f} m < 최소 이격 {cfg.min_clearance_m} m — "
                f"GSD 완화 또는 버퍼 조정 필요")

        # ① 시설물별 라인 설계 → ② 서펜타인 정렬
        lines = self.design_lines(surf, trig, line)
        res.n_lines = len(lines)
        pts, line_idx = [], []
        for i, ln in enumerate(lines):
            seq = ln[::-1] if i % 2 == 1 else ln      # 홀수 라인 역방향
            pts.append(seq)
            line_idx += [i] * len(seq)
        pts = np.vstack(pts)
        res.line_index = np.array(line_idx)

        # ③ 법선·짐벌각·웨이포인트
        solver = TiltSolver(approach_dist_m=d, fit_radius_m=cfg.fit_radius_m,
                            smooth_window=cfg.smooth_window,
                            max_pitch_rate=cfg.max_pitch_rate,
                            gimbal_range=cfg.gimbal_range)
        fit_v = (mesh_vertices[surf.vertex_idx] if getattr(surf, "vertex_idx", None)
                 is not None else mesh_vertices)      # 가장자리에서 인접 지형 혼입 방지
        sol = solver.solve(pts, fit_v, surf.mean_normal)
        res.waypoints = sol.waypoints
        res.warnings += solver.warnings

        # ④⑤ 안전검사·소티 분할·통계 (신규 생성/기존 적응 공용)
        return self.finalize(res, mesh_vertices)

    def finalize(self, res: PathResult, mesh_vertices: np.ndarray) -> PathResult:
        """경로 확정 공통 단계: 최소이격 검사 → 소티 분할 → 데이터량."""
        if not res.waypoints:
            return res
        if res.line_index is None:
            res.line_index = np.zeros(len(res.waypoints), dtype=int)
        self._check_clearance(res, mesh_vertices)
        self._split_sorties(res)
        res.data_gb = len(res.waypoints) * self.cam.mb_per_photo / 1024.0
        return res

    # ── 안전검사 ──
    def _check_clearance(self, res: PathResult, mesh_vertices: np.ndarray):
        tree = cKDTree(mesh_vertices)
        dist, _ = tree.query(res.positions())
        bad = np.where(dist < self.cfg.min_clearance_m)[0]
        for i in bad[:10]:                              # 과다 출력 방지
            res.warnings.append(
                f"[이격 위반] WP#{i} 표면거리 {dist[i]:.1f} m "
                f"< {self.cfg.min_clearance_m} m")
        if len(bad) > 10:
            res.warnings.append(f"[이격 위반] 외 {len(bad)-10}건 추가")
        res.min_clearance_measured = float(dist.min())

    # ── 소티 분할 (누적 시간 기준, 라인 경계에서만 절단) ──
    def _split_sorties(self, res: PathResult):
        cfg = self.cfg
        pos = res.positions()
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        res.path_len_m = float(seg.sum())
        t_cum = np.concatenate([[0.0], np.cumsum(seg / cfg.speed_ms)])
        t_cum += np.arange(len(pos)) * cfg.wp_overhead_s   # WP 오버헤드
        res.flight_time_min = float(t_cum[-1] / 60.0)

        limit_s = cfg.safe_endurance_min * 60.0
        sortie = np.zeros(len(pos), dtype=int)
        s, t0 = 0, 0.0
        for i in range(1, len(pos)):
            if t_cum[i] - t0 > limit_s and res.line_index[i] != res.line_index[i - 1]:
                s += 1
                t0 = t_cum[i]                              # 라인 경계에서 절단
            sortie[i] = s
        res.sortie_index = sortie
        res.n_sorties = int(sortie.max()) + 1
