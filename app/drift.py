"""单点漂移（跳点）检测与修复。

判定与修复规则（对每个原始 trkseg 独立进行，不跨分段）：

- 前置校验：同一 trkseg 内相邻轨迹点时间必须严格递增；
  零时差或倒序时报出明确错误（ValueError），不做任何修复；
- 某中间点 P（前驱 A、后继 B）同时满足以下三条即判为跳点：
  1. A→P 与 P→B 两段速度均超过 12 米/秒；
  2. 绕过 P 的 A→B 首尾速度不超过 12 米/秒；
  3. P 到线段 A→B 的水平最短距离超过 40 米；
- 修复按原始点顺序逐轮进行：每轮只删除最早命中的一个跳点，
  删除后在剩余点上重新判定，直到无命中；每段首尾点始终保留
  （跳点判定只针对中间点，端点天然不会被删除）；
- 本模块只读取原始轨迹，不修改上传时保存的数据：
  遮蔽裁切、分段配速与海拔分析仍基于原始轨迹，互不影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geo import TrackPoint, haversine_m, point_segment_distance_m

MAX_SPEED_MPS = 12.0  # 判定跳点的速度阈值（米/秒）：超过才算异常
MIN_OFFSET_M = 40.0   # 判定跳点的偏离阈值（米）：到首尾线段的水平最短距离超过才算异常


@dataclass
class JumpHit:
    """一次命中（被删除）的跳点记录，速度/偏离为删除当轮的计算值。"""

    seg_index: int          # 原始 trkseg 序号（从 1 开始）
    point_index: int        # 被删点在原始 trkseg 中的序号（从 1 开始）
    round: int              # 该 trkseg 内的删除轮次（从 1 开始）
    point: TrackPoint
    speed_in_mps: float     # 前一段速度（A→P）
    speed_out_mps: float    # 后一段速度（P→B）
    speed_direct_mps: float # 绕过该点的首尾速度（A→B）
    offset_m: float         # P 到线段 A→B 的水平最短距离


@dataclass
class DriftResult:
    """漂移修复结果；fixed_segments 与原始结构一致（各段保留首尾点）。"""

    fixed_segments: list                  # list[list[TrackPoint]]
    hits: list = field(default_factory=list)  # list[JumpHit]，按删除顺序
    distance_before_m: float = 0.0
    distance_after_m: float = 0.0


def _speed_mps(a: TrackPoint, b: TrackPoint) -> float:
    """a→b 的水平速度（米/秒）；调用前须保证 b.time > a.time。"""
    return haversine_m(a, b) / (b.time - a.time).total_seconds()


def _find_earliest_jump(points: list):
    """在当前点序列中按原始顺序找最早命中的跳点。

    返回 (下标, 前段速度, 后段速度, 首尾速度, 偏离距离)，无命中返回 None。
    只检查中间点（下标 1..n-2），首尾点永不命中。
    """
    for k in range(1, len(points) - 1):
        a, p, b = points[k - 1], points[k], points[k + 1]
        v_in = _speed_mps(a, p)
        if v_in <= MAX_SPEED_MPS:
            continue
        v_out = _speed_mps(p, b)
        if v_out <= MAX_SPEED_MPS:
            continue
        v_direct = _speed_mps(a, b)
        if v_direct > MAX_SPEED_MPS:
            continue
        offset = point_segment_distance_m(p, a, b)
        if offset <= MIN_OFFSET_M:
            continue
        return k, v_in, v_out, v_direct, offset
    return None


def fix_drift(segments: list) -> DriftResult:
    """对整条轨迹（若干原始 trkseg）检测并修复单点漂移。

    每个 trkseg 独立处理：先校验相邻时间严格递增，再逐轮删除最早命中的
    跳点并重算，直到无命中。时间不递增时抛出 ValueError（由接口层转为提示）。
    """
    fixed_segments: list = []
    hits: list = []
    distance_before = 0.0
    distance_after = 0.0

    for seg_idx, seg in enumerate(segments, start=1):
        for i, (a, b) in enumerate(zip(seg, seg[1:]), start=1):
            distance_before += haversine_m(a, b)
            if b.time <= a.time:
                raise ValueError(
                    f"第 {seg_idx} 个 trkseg 中第 {i}→{i + 1} 个轨迹点的时间"
                    "未严格递增（零时差或倒序），无法进行漂移修复"
                )

        work = list(seg)                       # 工作副本：逐轮删除跳点
        orig_index = list(range(1, len(seg) + 1))  # 工作序列各点的原始序号（从 1 开始）
        round_no = 0
        while len(work) >= 3:
            found = _find_earliest_jump(work)
            if found is None:
                break
            k, v_in, v_out, v_direct, offset = found
            round_no += 1
            hits.append(JumpHit(
                seg_index=seg_idx,
                point_index=orig_index[k],
                round=round_no,
                point=work[k],
                speed_in_mps=v_in,
                speed_out_mps=v_out,
                speed_direct_mps=v_direct,
                offset_m=offset,
            ))
            del work[k]
            del orig_index[k]

        for a, b in zip(work, work[1:]):
            distance_after += haversine_m(a, b)
        fixed_segments.append(work)

    return DriftResult(
        fixed_segments=fixed_segments,
        hits=hits,
        distance_before_m=distance_before,
        distance_after_m=distance_after,
    )
