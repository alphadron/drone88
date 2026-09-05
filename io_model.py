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
