#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 path_adapter.py — FacilityPath v1.1 / [4'단계] 기존 경로 → 시설물 형상 적응
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · DID-DP-2026-0829 부속 (2026-09-05)
--------------------------------------------------------------------------------
 목적:
   UgCS 평면 그리드(연직) 경로나 수직 파사드 경로처럼 "형상을 모르고 만든"
   경로를, 기준면(3D 모델/DSM)에 맞춰 법선 정렬 경로로 바꾼다.
 알고리즘 (웨이포인트별, 순서 보존):
   ① 기준면에서 가장 가까운 점 p_s 와 국부 법선 n 을 찾는다
   ② 새 위치 = p_s + n·d   (d = 목표 접근거리, 또는 keep_distance면 원래 이격)
   ③ 짐벌 피치 φ = α − 90°, 헤딩 = 면을 향함 (TiltSolver)
   ④ 원래 경로에서 기준면과 너무 먼 점(off-surface, > max_snap_m)은 제외
   ⑤ 인접 웨이포인트 재배치 후 간격이 과밀(<min_gap_m)해지면 병합
 결과는 PathResult 로 반환되어 신규 생성 경로와 같은 출력 단계를 탄다.
================================================================================
"""

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from planner_base import PathResult, PlanConfig
from tilt_solver import TiltSolver, Waypoint
from camera import CameraSpec


def closest_on_mesh(mesh: trimesh.Trimesh, pts: np.ndarray, k: int = 16):
    """
    각 점의 메시 최근접점. rtree 의존 없이 면 중심 KD-tree로 후보 k개를 뽑고
    trimesh.triangles.closest_point로 정확한 점-삼각형 거리를 계산한다.
    반환: (closest (N,3), dist (N,), face_id (N,))
    """
    tree = cKDTree(mesh.triangles_center)
    k = min(k, len(mesh.faces))
    _, cand = tree.query(pts, k=k)
    cand = np.atleast_2d(cand)
    tris = mesh.triangles[cand.ravel()]                       # (N*k,3,3)
    rep = np.repeat(pts, k, axis=0)                           # (N*k,3)
    cp = trimesh.triangles.closest_point(tris, rep).reshape(len(pts), k, 3)
    d = np.linalg.norm(cp - pts[:, None, :], axis=2)          # (N,k)
    j = d.argmin(axis=1)
    idx = np.arange(len(pts))
    return cp[idx, j], d[idx, j], cand[idx, j]


@dataclass
class AdaptConfig:
    approach_dist_m: float | None = None   # None → GSD로 산출 (PlanConfig.gsd_m)
    keep_distance: bool = False            # True: 원래 이격 유지, 법선만 정렬
    max_snap_m: float = 80.0               # 기준면에서 이보다 멀면 제외
    min_gap_m: float = 0.5                 # 재배치 후 과밀 병합 임계
    line_break_deg: float = 60.0           # 진행방향 급변(>이 각) → 새 라인으로 간주


class PathAdapter:
    def __init__(self, camera: CameraSpec, plan_cfg: PlanConfig,
                 adapt_cfg: AdaptConfig = None):
        self.cam, self.cfg = camera, plan_cfg
        self.acfg = adapt_cfg or AdaptConfig()

    # ── 라인 분할 (원래 경로의 꺾임으로 판단) ──
    def _line_index(self, pts: np.ndarray) -> np.ndarray:
        idx = np.zeros(len(pts), dtype=int)
        if len(pts) < 3:
            return idx
        d = np.diff(pts, axis=0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        cosang = np.einsum("ij,ij->i", d[:-1], d[1:])
        turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        line = 0
        for i, t in enumerate(turn, start=1):
            if t > self.acfg.line_break_deg:
                line += 1
            idx[i] = line
        idx[-1] = line
        return idx

    def adapt(self, enu_pts: np.ndarray, mesh: trimesh.Trimesh) -> PathResult:
        a, cfg = self.acfg, self.cfg
        d_target = a.approach_dist_m or self.cam.dist_for_gsd(cfg.gsd_m)
        res = PathResult(approach_dist_m=d_target)

        # ① 최근접 면상점·면 법선 (rtree 불필요한 자체 구현)
        closest, dist, fid = closest_on_mesh(mesh, enu_pts)
        keep = dist <= a.max_snap_m
        dropped = int((~keep).sum())
        if dropped:
            res.warnings.append(f"[적응] 기준면에서 {a.max_snap_m} m 초과 이격 {dropped}점 제외")
        pts, closest, dist, fid = enu_pts[keep], closest[keep], dist[keep], fid[keep]
        if len(pts) == 0:
            res.warnings.append("[적응] 유효 웨이포인트 없음"); return res

        # ② 국부 법선(SVD, 폴백=면 법선) → TiltSolver 공용 로직
        fn = mesh.face_normals[fid]
        fn[fn[:, 2] < 0] *= -1                                    # 상향 일관화
        solver = TiltSolver(approach_dist_m=d_target, fit_radius_m=cfg.fit_radius_m,
                            smooth_window=cfg.smooth_window,
                            max_pitch_rate=cfg.max_pitch_rate,
                            gimbal_range=cfg.gimbal_range)
        normals = solver.smooth_normals(
            solver.local_normals(closest, mesh.vertices, fn.mean(axis=0)))
        # 국부 법선이 원래 카메라 방향과 반대면(뒷면) 면 법선으로 대체
        toward = pts - closest
        flip = np.einsum("ij,ij->i", normals, toward) < 0
        normals[flip] = fn[flip]

        # ③ 재배치 + 짐벌각
        dists = dist if a.keep_distance else np.full(len(pts), d_target)
        new_pos = closest + normals * dists[:, None]

        # ⑤ 과밀 병합
        sel = [0]
        for i in range(1, len(new_pos)):
            if np.linalg.norm(new_pos[i] - new_pos[sel[-1]]) >= a.min_gap_m:
                sel.append(i)
        merged = len(new_pos) - len(sel)
        if merged:
            res.warnings.append(f"[적응] 재배치 후 과밀 {merged}점 병합")
        sel = np.array(sel)
        new_pos, closest, normals = new_pos[sel], closest[sel], normals[sel]

        wps, phis = [], []
        for p, s, n in zip(new_pos, closest, normals):
            phi, psi, alpha = solver.pitch_heading_from_normal(n)
            phis.append(phi)
            wps.append(Waypoint(position=p, surface_point=s, normal=n,
                                gimbal_pitch_deg=float(phi), heading_deg=float(psi),
                                slope_deg=float(alpha)))
        phis = solver.limit_pitch_rate(np.array(phis))
        for w, phi in zip(wps, phis):
            w.gimbal_pitch_deg = float(phi)
            if not (cfg.gimbal_range[0] <= phi <= cfg.gimbal_range[1]):
                res.warnings.append(f"[짐벌 한계] φ={phi:.1f}° at {np.round(w.position,1)}")

        res.waypoints = wps
        res.line_index = self._line_index(new_pos)
        res.n_lines = int(res.line_index.max()) + 1
        res.trigger_spacing_m = float(np.median(np.linalg.norm(np.diff(new_pos, axis=0), axis=1))) if len(new_pos) > 1 else 0.0
        return res
