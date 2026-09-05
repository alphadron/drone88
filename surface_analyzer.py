#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 surface_analyzer.py — FacilityPath v1.0 / M2 코어 모듈
 3D 메시에서 촬영 대상면을 추출하고 경사각(α)·주향(aspect)을 산출
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · 문서번호 DID-DP-2026-0829 부속 (M2)
--------------------------------------------------------------------------------
 좌표 규약 : ENU (x=East, y=North, z=Up), 단위 m
 경사각 α  : 수평면 기준 0°(평지) ~ 90°(수직벽) = arccos(n_z)
 주향 aspect: 경사면이 바라보는(내리막) 방위각 [0~360°, 북=0, 동=90]
             = 상향 법선의 수평성분 방위각
================================================================================
"""

from dataclasses import dataclass, field

import numpy as np
import trimesh


# ──────────────────────────────────────────────────────────────────────────────
# 결과 컨테이너
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SurfaceInfo:
    """추출된 촬영 대상면 1개의 요약 정보."""
    face_idx: np.ndarray = None       # 대상면을 구성하는 face 인덱스
    vertex_idx: np.ndarray = None     # 대상면 정점 인덱스 (국부 법선 피팅 범위 제한용)
    mean_slope_deg: float = 0.0       # 평균 경사각 α [deg]
    mean_aspect_deg: float = 0.0      # 평균 주향(내리막 방위각) [deg]
    mean_normal: np.ndarray = None    # 면적 가중 평균 법선 (상향, 단위벡터)
    area_m2: float = 0.0              # 면적 [m²]
    centroid: np.ndarray = None       # 면적 가중 중심 [m]
    extent_u_m: float = 0.0           # 주향직교(수평) 방향 길이 [m]
    extent_v_m: float = 0.0           # 경사 방향 길이 [m]

    def __repr__(self):
        return (f"SurfaceInfo(faces={len(self.face_idx):,}, "
                f"α={self.mean_slope_deg:.1f}°, aspect={self.mean_aspect_deg:.1f}°, "
                f"area={self.area_m2:,.1f} m², "
                f"extent={self.extent_u_m:.1f}×{self.extent_v_m:.1f} m)")


@dataclass
class AnalysisResult:
    surfaces: list = field(default_factory=list)   # SurfaceInfo 목록 (면적 내림차순)
    slope_hist: tuple = None                       # (bin_edges, counts) 경사각 분포


# ──────────────────────────────────────────────────────────────────────────────
# 기하 유틸
# ──────────────────────────────────────────────────────────────────────────────
def ensure_upward(normals: np.ndarray) -> np.ndarray:
    """법선을 상향(n_z ≥ 0)으로 정규화. 수직벽(n_z≈0)은 그대로 둔다."""
    n = normals.copy()
    flip = n[:, 2] < 0
    n[flip] *= -1.0
    return n


def slope_deg_of(normals: np.ndarray) -> np.ndarray:
    """상향 법선 → 경사각 α [deg]."""
    nz = np.clip(normals[:, 2], -1.0, 1.0)
    return np.degrees(np.arccos(nz))


def aspect_deg_of(normals: np.ndarray) -> np.ndarray:
    """상향 법선 → 주향(내리막 방위각) [deg, 북=0 동=90]. 평지(수평성분≈0)는 NaN."""
    east, north = normals[:, 0], normals[:, 1]
    horiz = np.hypot(east, north)
    asp = np.degrees(np.arctan2(east, north)) % 360.0
    asp[horiz < 1e-9] = np.nan
    return asp


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray = None) -> float:
    """방위각의 원형 평균 [deg]."""
    a = np.radians(angles_deg[~np.isnan(angles_deg)])
    if a.size == 0:
        return float("nan")
    w = np.ones_like(a) if weights is None else weights[~np.isnan(angles_deg)]
    s, c = np.average(np.sin(a), weights=w), np.average(np.cos(a), weights=w)
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 분석기
# ──────────────────────────────────────────────────────────────────────────────
class SurfaceAnalyzer:
    """
    메시에서 경사각 범위로 촬영 대상면을 추출한다.

    파이프라인:
      ① 면 법선 상향 정규화 → 경사각/주향 산출
      ② 경사각 범위 필터 (slope_min ≤ α ≤ slope_max)
      ③ 인접성 기반 연결요소 분리 (면 단위 region)
      ④ 소면적 노이즈 제거(min_area_m2) 후 면적 내림차순 정렬
    """

    def __init__(self, slope_min_deg: float = 25.0, slope_max_deg: float = 90.0,
                 min_area_m2: float = 10.0):
        self.slope_min = slope_min_deg
        self.slope_max = slope_max_deg
        self.min_area = min_area_m2

    # ── 로드 ──
    @staticmethod
    def load_mesh(path: str) -> trimesh.Trimesh:
        """OBJ/PLY/STL 등 로드. Scene이면 단일 메시로 병합."""
        m = trimesh.load(path, force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            raise ValueError(f"메시 로드 실패: {path}")
        return m

    # ── 분석 ──
    def analyze(self, mesh: trimesh.Trimesh) -> AnalysisResult:
        fn = ensure_upward(mesh.face_normals)
        slope = slope_deg_of(fn)
        aspect = aspect_deg_of(fn)
        area = mesh.area_faces

        # 경사각 분포 (UI 히스토그램용)
        hist = np.histogram(slope, bins=18, range=(0, 90), weights=area)

        # ① 경사각 범위 필터
        cand = np.where((slope >= self.slope_min) & (slope <= self.slope_max))[0]
        result = AnalysisResult(slope_hist=hist)
        if cand.size == 0:
            return result

        # ② 후보면들만의 서브메시에서 연결요소 분리
        sub = mesh.submesh([cand], append=True)
        labels = trimesh.graph.connected_component_labels(sub.face_adjacency,
                                                          node_count=len(sub.faces))
        sub_fn = ensure_upward(sub.face_normals)
        sub_slope = slope_deg_of(sub_fn)
        sub_aspect = aspect_deg_of(sub_fn)
        sub_area = sub.area_faces
        sub_centers = sub.triangles_center

        for lb in np.unique(labels):
            idx = np.where(labels == lb)[0]
            a = sub_area[idx]
            if a.sum() < self.min_area:          # ④ 노이즈 제거
                continue
            n_mean = np.average(sub_fn[idx], axis=0, weights=a)
            n_mean /= np.linalg.norm(n_mean)
            info = SurfaceInfo(
                face_idx=cand[idx],
                vertex_idx=np.unique(mesh.faces[cand[idx]]),
                mean_slope_deg=float(np.average(sub_slope[idx], weights=a)),
                mean_aspect_deg=circular_mean_deg(sub_aspect[idx], a),
                mean_normal=n_mean,
                area_m2=float(a.sum()),
                centroid=np.average(sub_centers[idx], axis=0, weights=a),
            )
            self._fit_extent(info, sub_centers[idx])
            result.surfaces.append(info)

        result.surfaces.sort(key=lambda s: -s.area_m2)
        return result

    @staticmethod
    def _fit_extent(info: SurfaceInfo, pts: np.ndarray):
        """면의 2D 파라미터 축(u=주향직교 수평, v=경사) 방향 범위 산출."""
        n = info.mean_normal
        up = np.array([0.0, 0.0, 1.0])
        u = np.cross(up, n)                      # 수평·면내 방향 (등고선 방향)
        if np.linalg.norm(u) < 1e-9:             # 평지: 임의 수평축
            u = np.array([1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)                       # 경사(오르막) 방향, 면내
        d = pts - info.centroid
        info.extent_u_m = float(np.ptp(d @ u))
        info.extent_v_m = float(np.ptp(d @ v))


# ──────────────────────────────────────────────────────────────────────────────
# 합성 경사면 생성기 (검증·데모용)
# ──────────────────────────────────────────────────────────────────────────────
def make_synthetic_slope(slope_deg: float, aspect_deg: float = 180.0,
                         width_m: float = 60.0, length_m: float = 30.0,
                         grid: int = 24, noise_m: float = 0.0,
                         seed: int = 42) -> trimesh.Trimesh:
    """
    경사각 slope_deg, 주향 aspect_deg(내리막 방위각)의 평면 경사 메시 생성.
    width_m: 등고선(u) 방향 폭, length_m: 경사(v) 방향 사면 길이(빗변 기준).
    noise_m: 표면 거칠기(법선 방향 랜덤 변위) — 국부 법선 추정 강건성 시험용.
    """
    a = np.radians(slope_deg)
    b = np.radians(aspect_deg)
    # 면내 축: u=등고선 방향(수평), v=오르막 방향
    u_ax = np.array([np.cos(b), -np.sin(b), 0.0])                    # aspect에 직교(수평)
    v_ax = np.array([-np.sin(b) * np.cos(a), -np.cos(b) * np.cos(a), np.sin(a)])
    uu, vv = np.meshgrid(np.linspace(-width_m / 2, width_m / 2, grid),
                         np.linspace(0, length_m, grid))
    pts = uu[..., None] * u_ax + vv[..., None] * v_ax
    if noise_m > 0:
        rng = np.random.default_rng(seed)
        n_ax = np.cross(u_ax, v_ax)
        pts = pts + rng.normal(0, noise_m, uu.shape)[..., None] * n_ax
    verts = pts.reshape(-1, 3)
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            k = i * grid + j
            faces += [[k, k + 1, k + grid], [k + 1, k + grid + 1, k + grid]]
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


if __name__ == "__main__":
    # 간이 자가시험: 60° 남향 사면
    m = make_synthetic_slope(60.0, 180.0)
    res = SurfaceAnalyzer(slope_min_deg=10).analyze(m)
    for s in res.surfaces:
        print(s)
