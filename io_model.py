#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 io_model.py — FacilityPath v1.1 / [1단계] 기준면 입력
 3D 메시(OBJ/PLY/STL) 또는 DSM 래스터(GeoTIFF/ASC/XYZ) → 로컬 ENU Trimesh
--------------------------------------------------------------------------------
 (주)드론아이디 AI연구소 · DID-DP-2026-0829 부속 (M2/M4 보강, 2026-09-05)
--------------------------------------------------------------------------------
 입력 유형:
   A) 메시   : iTwin Capture Modeler 산출 OBJ/PLY/STL  (좌표: 로컬 또는 투영좌표)
   B) DSM    : GeoTIFF(.tif, rasterio 필요) / ESRI ASCII(.asc) / XYZ 텍스트
               → 격자 삼각화로 메시 생성 (경로 설계용 데시메이션 step 지원)
 좌표 규약:
   내부 계산은 항상 로컬 ENU(m). 입력이 투영좌표(EPSG:5186 등)이면 기준점을
   빼서 ENU로 옮기고, 기준점의 경위도를 pyproj로 구해 GeoAnchor로 보관한다.
   → 이후 kml_writer가 같은 GeoAnchor로 WGS84 복원.
================================================================================
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import trimesh
from pyproj import CRS, Transformer

from kml_writer import GeoAnchor


@dataclass
class ReferenceModel:
    mesh: trimesh.Trimesh          # 로컬 ENU 메시
    anchor: GeoAnchor              # ENU 원점의 WGS84 위치
    source: str                    # 입력 파일
    kind: str                      # "mesh" | "dsm"
    epsg: int | None = None        # 입력 투영좌표계 (로컬이면 None)
    origin_xy: tuple | None = None # 투영좌표 기준점 (x0, y0)


# ──────────────────────────────────────────────────────────────────────────────
# 기준점 유틸
# ──────────────────────────────────────────────────────────────────────────────
def projected_to_anchor(x0: float, y0: float, z0: float, epsg: int) -> GeoAnchor:
    """투영좌표 기준점 → WGS84 GeoAnchor."""
    tr = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
    lon, lat = tr.transform(x0, y0)
    return GeoAnchor(lat=float(lat), lon=float(lon), h=float(z0))


def _to_enu(vertices: np.ndarray, epsg, anchor: GeoAnchor | None):
    """정점을 로컬 ENU로. 반환 (enu, anchor, origin_xy)."""
    if epsg is None:                                  # 이미 로컬 좌표
        if anchor is None:
            raise ValueError("로컬 좌표 모델은 --anchor lat,lon,h 가 필요합니다")
        return vertices.copy(), anchor, None
    x0, y0 = float(vertices[:, 0].mean()), float(vertices[:, 1].mean())
    z0 = float(vertices[:, 2].min())
    enu = vertices - np.array([x0, y0, z0])
    return enu, projected_to_anchor(x0, y0, z0, epsg), (x0, y0)


# ──────────────────────────────────────────────────────────────────────────────
# A) 메시 로더
# ──────────────────────────────────────────────────────────────────────────────
def load_mesh_model(path: str, epsg: int | None = None,
                    anchor: GeoAnchor | None = None,
                    max_faces: int = 500_000) -> ReferenceModel:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"메시 로드 실패: {path}")
    if len(m.faces) > max_faces:                      # 경로 설계용 데시메이션
        try:
            m = m.simplify_quadric_decimation(max_faces)
        except Exception:
            pass                                      # fast_simplification 미설치 시 원본 유지
    enu, anc, org = _to_enu(np.asarray(m.vertices, float), epsg, anchor)
    mesh = trimesh.Trimesh(vertices=enu, faces=m.faces, process=True)
    return ReferenceModel(mesh, anc, path, "mesh", epsg, org)


# ──────────────────────────────────────────────────────────────────────────────
# B) DSM 로더
# ──────────────────────────────────────────────────────────────────────────────
def _read_asc(path: str):
    """ESRI ASCII Grid → (xs, ys, Z, nodata)."""
    hdr = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and lines[i].split()[0].lower() in (
            "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
            "yllcenter", "cellsize", "nodata_value"):
        k, v = lines[i].split()[:2]
        hdr[k.lower()] = float(v); i += 1
    Z = np.loadtxt(lines[i:], dtype=float)
    nc, nr, cs = int(hdr["ncols"]), int(hdr["nrows"]), hdr["cellsize"]
    x0 = hdr.get("xllcorner", hdr.get("xllcenter", 0) - cs / 2) + cs / 2
    y0 = hdr.get("yllcorner", hdr.get("yllcenter", 0) - cs / 2) + cs / 2
    xs = x0 + cs * np.arange(nc)
    ys = y0 + cs * np.arange(nr)[::-1]               # ASC는 북→남 순서
    return xs, ys, Z, hdr.get("nodata_value", -9999.0)


def _read_xyz(path: str):
    """정규 격자 XYZ 텍스트 → (xs, ys, Z, nodata). 열: x y z (구분자 공백/콤마)."""
    d = np.loadtxt(path, delimiter=None if open(path).readline().count(",") == 0 else ",")
    xs, ys = np.unique(d[:, 0]), np.unique(d[:, 1])
    Z = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, d[:, 0]); iy = np.searchsorted(ys, d[:, 1])
    Z[iy, ix] = d[:, 2]
    return xs, ys, Z, np.nan


def _read_geotiff(path: str):
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("GeoTIFF DSM에는 rasterio가 필요합니다: pip install rasterio") from e
    with rasterio.open(path) as ds:
        Z = ds.read(1).astype(float)
        t = ds.transform
        xs = t.c + t.a * (np.arange(ds.width) + 0.5)
        ys = t.f + t.e * (np.arange(ds.height) + 0.5)
        nodata = ds.nodata if ds.nodata is not None else np.nan
        epsg = ds.crs.to_epsg() if ds.crs else None
    return xs, ys, Z, nodata, epsg


def dsm_to_mesh(xs, ys, Z, nodata, step: int = 1) -> trimesh.Trimesh:
    """격자 DSM → 삼각 메시(투영좌표 그대로). nodata 셀은 제외."""
    xs, ys, Z = xs[::step], ys[::step], Z[::step, ::step]
    valid = ~(np.isnan(Z) | (Z == nodata))
    X, Y = np.meshgrid(xs, ys)
    nr, nc = Z.shape
    idx = -np.ones((nr, nc), dtype=int)
    idx[valid] = np.arange(valid.sum())
    verts = np.column_stack([X[valid], Y[valid], Z[valid]])
    faces = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            if min(a, b, c, d) < 0:
                continue
            faces += [[a, b, c], [b, d, c]]
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


def load_dsm_model(path: str, epsg: int | None = None, step: int = 1) -> ReferenceModel:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        xs, ys, Z, nodata, epsg_file = _read_geotiff(path)
        epsg = epsg or epsg_file
    elif ext == ".asc":
        xs, ys, Z, nodata = _read_asc(path)
    else:
        xs, ys, Z, nodata = _read_xyz(path)
    if epsg is None:
        raise ValueError("DSM의 좌표계 EPSG를 지정하십시오 (예: --epsg 5186)")
    m = dsm_to_mesh(xs, ys, Z, nodata, step)
    enu, anc, org = _to_enu(np.asarray(m.vertices, float), epsg, None)
    mesh = trimesh.Trimesh(vertices=enu, faces=m.faces, process=True)
    return ReferenceModel(mesh, anc, path, "dsm", epsg, org)


# ──────────────────────────────────────────────────────────────────────────────
# C) 경계 다각형(Google Earth Pro 실측 평면 범위) → 배터 사면 자동 구성
# ──────────────────────────────────────────────────────────────────────────────
def _fit_baseline_pca(boundary_en: np.ndarray):
    """다각형 정점 전체의 수평 주성분(PCA)으로 기준선(진행) 방향을 구한다.
    도로 굽이 등으로 경계가 여러 변으로 꺾여 있어도(개별 변 하나만 보는
    것보다) 강건하게 전체적인 진행방향을 잡는다.
    반환: (단위방향 u2, 중심점 C)."""
    C = boundary_en.mean(axis=0)
    _, _, Vt = np.linalg.svd(boundary_en - C, full_matrices=False)
    return Vt[0], C


def _rise_dir(u2: np.ndarray, clockwise: bool) -> np.ndarray:
    """기준선 방향 u2를 평면도(위에서 내려다본 지도)에서 90° 회전시켜
    사면이 올라가는 쪽(오르막 수평방향)을 구한다.
      clockwise=True (경사비 부호 +): 시계방향 90° → 기준선 진행방향의 오른쪽
      clockwise=False(경사비 부호 −): 반시계방향 90° → 진행방향의 왼쪽
    """
    return np.array([u2[1], -u2[0]]) if clockwise else np.array([-u2[1], u2[0]])


def _refine_baseline_on_base_edge(boundary_en: np.ndarray, u2: np.ndarray,
                                  rise2: np.ndarray, band_frac: float = 0.3):
    """기준선을 다각형의 "밑변"(오르막 반대쪽 변)에 맞춘다.
    오르막 방향 투영값이 하위 band_frac 구간에 드는 정점(=밑변을 이루는
    정점들)만 골라 그 정점들의 주성분으로 기준선 방향을 다시 잡는다.
    전체 PCA는 굽은 다각형에서 밑변이 아니라 전체 형상의 평균 방향을 주므로,
    "경사면 밑변을 평면 다각형의 밑변에 일치"시키려면 이 보정이 필요하다.
    정점이 2개 미만이면 원래 방향을 그대로 돌려준다."""
    t = boundary_en @ rise2
    t_lo, t_hi = float(t.min()), float(t.max())
    if t_hi - t_lo < 1e-9:
        return u2
    sel = boundary_en[t <= t_lo + band_frac * (t_hi - t_lo)]
    if len(sel) < 2:
        return u2
    u_ref, _ = _fit_baseline_pca(sel)
    if float(u_ref @ u2) < 0:            # 진행방향 부호는 기존과 맞춘다
        u_ref = -u_ref
    return u_ref


def build_slope_from_boundary(boundary_en: np.ndarray, ratio_v: float, ratio_h: float,
                              baseline_azimuth_deg: float | None = None,
                              height_m: float | None = None,
                              margin_m: float = 5.0, grid: int = 8):
    """평면 경계 다각형(로컬 ENU E,N, 폐합 불필요) + 경사비(V:H) →
    기준선(밑변)을 축으로 회전시킨 사면 메시를 구성한다.

    개념(사면 = 기준선을 경첩으로 세운 평면):
      ① 기준선 = 평면 다각형의 "밑변" — 경사면의 밑변(toe)을 여기에 일치시킨다.
         방향은 다각형 정점의 수평 주성분(PCA)으로 잡되, 오르막 반대쪽
         (밑변) 정점들만 다시 PCA해 실제 밑변에 맞춘다.
         baseline_azimuth_deg[deg, 북=0 동=90]로 직접 지정할 수도 있다.
      ② 경사 방향 = 경사비의 부호. 평면도(위에서 본 지도) 기준으로
         **+(시계방향)**: 기준선 진행방향의 **오른쪽**이 올라간다.
         **−(반시계방향)**: 기준선 진행방향의 **왼쪽**이 올라간다.
         (예: ratio_v=+1, ratio_h=0.3 → 1:0.3 시계방향 / ratio_v=−1 → 반시계)
      ③ 기준선은 오르막의 반대쪽 극단(=밑변)에 놓이고, 사면은 거기서
         다각형 전체 폭(수평런)만큼 올라간다 → 기준면이 항상 다각형 전체를
         덮는다.

    사면 높이는 수평런 × |V|/H 로 역산하되, height_m을 지정하면 그 값을
    그대로 쓴다(평면도가 부정확할 때 보정용).

    반환: (mesh: 로컬 ENU Trimesh, info: dict — alpha_deg, aspect_deg,
    baseline_azimuth_deg, rotation, width_m, run_m, height_m, height_source)
    """
    clockwise = ratio_v >= 0
    ratio_v = abs(ratio_v)

    if baseline_azimuth_deg is not None:
        b = math.radians(baseline_azimuth_deg)
        u2 = np.array([math.sin(b), math.cos(b)])   # 방위각→(E,N), surface_analyzer 규약과 동일
    else:
        u2, _ = _fit_baseline_pca(boundary_en)
        # 오르막 반대쪽(밑변) 정점들로 기준선 방향을 한 번 더 맞춘다
        u2 = _refine_baseline_on_base_edge(boundary_en, u2, _rise_dir(u2, clockwise))

    rise2 = _rise_dir(u2, clockwise)                # 오르막 수평방향(부호로 결정)
    proj_u = boundary_en @ u2
    proj_r = boundary_en @ rise2

    r_lo, r_hi = float(proj_r.min()), float(proj_r.max())
    run_m = r_hi - r_lo                              # 밑변 → 최상단까지 수평런
    if run_m < 1e-6:
        raise ValueError("경계 다각형이 기준선 위에 퇴화되어 있습니다 — "
                         "면적을 가진 다각형인지 확인하십시오")

    alpha_deg = math.degrees(math.atan(ratio_v / ratio_h))
    if height_m is None:
        height_m = run_m * (ratio_v / ratio_h)
        height_source = "auto(다각형 폭 역산)"
    else:
        height_source = "지정값(--slope-height)"

    u_lo, u_hi = float(proj_u.min()) - margin_m, float(proj_u.max()) + margin_m
    width_m = u_hi - u_lo
    u_mid = (u_lo + u_hi) / 2
    length_m = height_m / math.sin(math.radians(alpha_deg))

    u_ax = np.array([u2[0], u2[1], 0.0])
    # 기준선(경첩)을 축으로 회전 → 오르막 수평성분 cosα, 수직성분 sinα
    v_ax = np.array([rise2[0] * math.cos(math.radians(alpha_deg)),
                     rise2[1] * math.cos(math.radians(alpha_deg)),
                     math.sin(math.radians(alpha_deg))])
    # 밑변(오르막 반대쪽 극단) 위의 점 — 경사면 밑변을 다각형 밑변에 일치
    base_xy = u_mid * u2 + r_lo * rise2
    base = np.array([base_xy[0], base_xy[1], 0.0])

    uu, vv = np.meshgrid(np.linspace(-width_m / 2, width_m / 2, grid),
                        np.linspace(0.0, length_m, grid))
    verts = (base + uu[..., None] * u_ax + vv[..., None] * v_ax).reshape(-1, 3)
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            k = i * grid + j
            faces += [[k, k + 1, k + grid], [k + 1, k + grid + 1, k + grid]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)

    # 사면 법선의 수평 성분은 -rise2(내리막) 방향을 향한다(rise2는 오르막)
    # — surface_analyzer의 aspect 규약(내리막 방위각)과 맞추려 부호 반전.
    info = dict(alpha_deg=alpha_deg,
               aspect_deg=math.degrees(math.atan2(-rise2[0], -rise2[1])) % 360,
               baseline_azimuth_deg=math.degrees(math.atan2(u2[0], u2[1])) % 360,
               rotation="시계방향(+)" if clockwise else "반시계방향(−)",
               rise_azimuth_deg=math.degrees(math.atan2(rise2[0], rise2[1])) % 360,
               width_m=width_m, run_m=run_m, height_m=height_m,
               height_source=height_source)
    return mesh, info


# ──────────────────────────────────────────────────────────────────────────────
# 통합 진입점
# ──────────────────────────────────────────────────────────────────────────────
def load_reference(path: str, epsg: int | None = None,
                   anchor: GeoAnchor | None = None, dsm_step: int = 1) -> ReferenceModel:
    """확장자로 메시/DSM을 자동 판별해 로컬 ENU ReferenceModel 반환."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".obj", ".ply", ".stl", ".off", ".glb", ".gltf"):
        return load_mesh_model(path, epsg, anchor)
    if ext in (".tif", ".tiff", ".asc", ".xyz", ".txt", ".csv"):
        return load_dsm_model(path, epsg, dsm_step)
    raise ValueError(f"지원하지 않는 입력 형식: {ext}")
