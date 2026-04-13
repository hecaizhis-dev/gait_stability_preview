"""
stu_raw CSV 列名适配 + 加速度语义统一（线性加速度 vs 含重力加速度）。

输出列名需满足 algorithm/M1/features_extract.py 的 data_load 读取要求：
  Timestamp, ACC_X_SA, ACC_Y_SA, ACC_Z_SA, GYR_X_SA, GYR_Y_SA, GYR_Z_SA

加速度语义（与力学约定一致）：
  a_with_gravity = a_linear + g
  a_linear       = a_with_gravity - g
其中 g 为重力在 **SA 传感器坐标系** 下的向量 (ACC_X_SA, ACC_Y_SA, ACC_Z_SA 对应分量)，
单位 m/s²。无个体标定时可在 config/default.yaml 的 imu_accel_adapter.gravity_vector_sa_mps2 中调整。

为何用列名推断 + 可选覆盖：
  仅凭数值无法可靠区分 linear / with_gravity（需姿态）；采集端列名常带
  Linear Acceleration / Accelerometer 等语义，故优先从列名推断，必要时用 input_override 强制。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.loader import load_config

# features_extract 在 mobgap 之前需要的列
REQUIRED_COLUMNS = frozenset(
    {
        "Timestamp",
        "ACC_X_SA",
        "ACC_Y_SA",
        "ACC_Z_SA",
        "GYR_X_SA",
        "GYR_Y_SA",
        "GYR_Z_SA",
    }
)

GYR_DEFAULT_ZERO_COLS = ["GYR_X_SA", "GYR_Y_SA", "GYR_Z_SA"]

# 默认与 config/default.yaml 中 imu_accel_adapter 一致（config 缺失时使用）
_DEFAULT_ACCEL_CFG = {
    "target": "with_gravity",
    "gravity_vector_sa_mps2": [0.0, 0.0, -9.80665],
    "input_override": "auto",
}


def _normalize_colname(name: object) -> str:
    s = str(name).strip().upper()
    s = s.replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _map_to_required(norm_col: str) -> Optional[str]:
    """将原始列名映射到标准列名（仅重命名，不改数值）。"""
    if norm_col in {"TIMESTAMP", "TIME", "TS"} or "TIMESTAMP" in norm_col or (
        norm_col.startswith("TIME") and len(norm_col) <= 12
    ):
        return "Timestamp"

    if "ACC" in norm_col:
        if "X" in norm_col:
            return "ACC_X_SA"
        if "Y" in norm_col:
            return "ACC_Y_SA"
        if "Z" in norm_col:
            return "ACC_Z_SA"

    if ("M/S" in norm_col) or ("M/S^2" in norm_col) or ("M/S2" in norm_col):
        if norm_col.startswith("X"):
            return "ACC_X_SA"
        if norm_col.startswith("Y"):
            return "ACC_Y_SA"
        if norm_col.startswith("Z"):
            return "ACC_Z_SA"
        if "LINEAR_ACCELERATION" in norm_col:
            parts = norm_col.split("_")
            if len(parts) > 2 and parts[2] in ("X", "Y", "Z"):
                return f"ACC_{parts[2]}_SA"

    if "ABSOLUTE_ACCELERATION" in norm_col:
        return None

    if ("GYR" in norm_col) or ("GYRO" in norm_col):
        if "X" in norm_col:
            return "GYR_X_SA"
        if "Y" in norm_col:
            return "GYR_Y_SA"
        if "Z" in norm_col:
            return "GYR_Z_SA"

    for rc in REQUIRED_COLUMNS:
        if norm_col == _normalize_colname(rc):
            return rc

    return None


def build_rename_map(columns: Iterable[object]) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    used_targets: set = set()
    for c in columns:
        old = str(c)
        target = _map_to_required(_normalize_colname(old))
        if not target or target in used_targets:
            continue
        rename_map[old] = target
        used_targets.add(target)
    return rename_map


def _infer_accel_input_kind_from_original_columns(columns: Iterable[object]) -> str:
    """
    从原始列名推断加速度语义：
    - linear：列名含 LINEAR、USER_ACCELERATION、或仅有 X/Y/Z+(m/s^2) 无「加速度计/原始」等标签
      （多数手机/Physics Toolbox 导出「仅三轴+m/s²」时常为线加速度；若你设备实为含重力，请用 input_override）
    - with_gravity：列名含 ACCELEROMETER（且非 LINEAR_*）、RAW ACCEL、或明确 WITHOUT_LINEAR 的原始计读数
    - unknown：仍无法归类（极少见）
    """
    has_linear = False
    has_grav_hint = False
    norms = [_normalize_colname(c) for c in columns]

    # 仅三轴 + m/s²、列名里没有任何 LINEAR/ACCELEROMETER/ACCELERATION 等长词：常见为线加速度导出
    axis_ms2_count = 0
    for norm in norms:
        if "M/S" not in norm and "M/S2" not in norm:
            continue
        # X_(M/S^2) / Y_(M/S^2) / Z_(M/S^2) 等
        if len(norm) >= 2 and norm[0] in "XYZ" and norm[1] in "_(":
            if not any(
                kw in norm
                for kw in (
                    "LINEAR",
                    "ACCELEROMETER",
                    "ACCELERATION",
                    "GRAVITY",
                    "RAW",
                    "MAG",
                    "GYR",
                    "GYRO",
                )
            ):
                axis_ms2_count += 1

    for norm in norms:
        if not any(k in norm for k in ("M/S", "ACC", "ACCEL", "LINEAR")):
            continue
        if "LINEAR" in norm:
            has_linear = True
        # Android: TYPE_LINEAR_ACCELERATION 等
        if "USER_ACCELERATION" in norm or "USER_ACCEL" in norm:
            has_linear = True
        # 含重力：原始加速度计 / 未去重力
        if "ACCELEROMETER" in norm and "LINEAR" not in norm:
            has_grav_hint = True
        if ("ACCELERATION" in norm) and ("LINEAR" not in norm):
            if "LINEAR_ACCELERATION" not in norm:
                has_grav_hint = True
        if "RAW" in norm and ("ACC" in norm or "ACCEL" in norm):
            has_grav_hint = True

    # 三列都是 X/Y/Z + m/s² 的「极简列名」→ 按惯例判为 linear（与 Physics Toolbox 等常见导出一致）
    if axis_ms2_count >= 3 and not has_linear and not has_grav_hint:
        return "linear"

    if has_linear and not has_grav_hint:
        return "linear"
    if has_grav_hint and not has_linear:
        return "with_gravity"
    if has_linear and has_grav_hint:
        return "unknown"
    return "unknown"


def _load_accel_adapter_cfg() -> Tuple[str, np.ndarray, str]:
    try:
        cfg = load_config("default.yaml")
        block = cfg.get("imu_accel_adapter") or {}
    except Exception:
        block = {}
    target = str(block.get("target", _DEFAULT_ACCEL_CFG["target"])).lower().strip()
    if target not in ("linear", "with_gravity"):
        target = str(_DEFAULT_ACCEL_CFG["target"])
    g = block.get("gravity_vector_sa_mps2", _DEFAULT_ACCEL_CFG["gravity_vector_sa_mps2"])
    g_arr = np.asarray(g, dtype=float).reshape(3)
    override = str(block.get("input_override", _DEFAULT_ACCEL_CFG["input_override"])).lower().strip()
    if override not in ("auto", "linear", "with_gravity", "unknown"):
        override = "auto"
    return target, g_arr, override


def _resolve_input_kind(inferred: str, override: str) -> str:
    """auto 用推断；否则用覆盖（unknown 表示强制不转换）。"""
    if override == "auto":
        return inferred
    if override == "unknown":
        return "unknown"
    return override


def _apply_accel_semantic_conversion(
    df: pd.DataFrame,
    input_kind: str,
    target_kind: str,
    gravity_sa: np.ndarray,
) -> pd.DataFrame:
    acc_cols = ["ACC_X_SA", "ACC_Y_SA", "ACC_Z_SA"]
    if not all(c in df.columns for c in acc_cols):
        return df
    if input_kind == "unknown" or input_kind == target_kind:
        return df
    if input_kind not in ("linear", "with_gravity") or target_kind not in ("linear", "with_gravity"):
        return df

    g = np.asarray(gravity_sa, dtype=float).reshape(3)
    out = df.copy()
    for i, c in enumerate(acc_cols):
        a = pd.to_numeric(out[c], errors="coerce").astype(float)
        if input_kind == "linear" and target_kind == "with_gravity":
            out[c] = a + g[i]
        elif input_kind == "with_gravity" and target_kind == "linear":
            out[c] = a - g[i]
    return out


def process_stu_raw_csv_columns(
    stu_raw_dir: str,
    stu_raw_modified_dir: str,
    encoding_read: str = "utf-8-sig",
    encoding_write: str = "utf-8-sig",
    strict: bool = False,
) -> None:
    """
    遍历 stu_raw 下所有 .csv，写入 stu_raw_modified：
    1) 列名映射到标准名；2) 按配置做 linear <-> with_gravity；3) 缺 GYR 时补 0。
    """
    in_base = Path(stu_raw_dir)
    out_base = Path(stu_raw_modified_dir)
    if not in_base.exists():
        raise FileNotFoundError(f"stu_raw_dir 不存在: {in_base}")
    out_base.mkdir(parents=True, exist_ok=True)

    accel_target, gravity_sa, input_override = _load_accel_adapter_cfg()

    for in_path in sorted(in_base.rglob("*.csv")):
        rel = in_path.relative_to(in_base)
        out_path = out_base / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(in_path, encoding=encoding_read)
        original_cols: List[str] = [str(x) for x in df.columns]
        inferred = _infer_accel_input_kind_from_original_columns(original_cols)
        input_kind = _resolve_input_kind(inferred, input_override)

        rename_map = build_rename_map(df.columns)
        if rename_map:
            df = df.rename(columns=rename_map)

        for col in GYR_DEFAULT_ZERO_COLS:
            if col not in df.columns:
                df[col] = 0.0

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing and strict:
            raise ValueError(
                f"{in_path} 缺少列: {sorted(missing)}；已有列: {list(df.columns)[:30]}..."
            )
        if missing and not strict:
            print(
                f"[WARN] {in_path.name} 缺少列 {sorted(missing)}（strict=False，仍写出当前列）"
            )

        # 有齐 ACC 三轴时才做语义转换
        if {"ACC_X_SA", "ACC_Y_SA", "ACC_Z_SA"}.issubset(df.columns):
            df = _apply_accel_semantic_conversion(df, input_kind, accel_target, gravity_sa)
            if input_kind == "unknown":
                print(
                    f"[ACCEL] {in_path.name}: 无法从列名判断 linear/with_gravity（inferred=unknown），"
                    f"未改加速度数值；请在 config/default.yaml 设置 imu_accel_adapter.input_override 为 linear 或 with_gravity"
                )
            elif input_kind != accel_target:
                print(
                    f"[ACCEL] {in_path.name}: input={input_kind} (inferred={inferred}, override={input_override}) "
                    f"-> target={accel_target}, g_sa={gravity_sa.tolist()}"
                )

        df.to_csv(out_path, index=False, encoding=encoding_write)
        print(f"[OK] {str(rel).replace(os.sep, '/')}")


if __name__ == "__main__":
    process_stu_raw_csv_columns(
        str(Path("data") / "dataset" / "stu_raw"),
        str(Path("data") / "dataset" / "stu_raw_modified"),
    )
