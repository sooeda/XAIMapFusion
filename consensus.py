from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic

from config import ALL_BASELINE_METHODS, Config
from utils import normalize01, selected_triplet_from_cfg


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


class ConsensusMixin:
    cfg: Config

    def normalize01(self, x):
        return normalize01(x)

    def positive_importance_map(self, pixel_map):
        pos = np.clip(np.asarray(pixel_map, dtype=np.float32), 0.0, None)
        mx = float(pos.max()) if pos.size else 0.0
        if mx <= 0:
            return np.zeros_like(pos, dtype=np.float32)
        return (pos / mx).astype(np.float32)

    def top_fraction_mask(self, pos_map, top_fraction=None):
        top_fraction = self.cfg.top_positive_fraction if top_fraction is None else top_fraction
        vals = pos_map[pos_map > 0]
        if len(vals) == 0:
            return np.zeros_like(pos_map, dtype=np.uint8)
        thr = np.quantile(vals, max(0.0, min(1.0, 1.0 - top_fraction)))
        return (pos_map >= thr).astype(np.uint8)

    def region_top_fraction_support(self, region_scores: np.ndarray, top_fraction=None) -> np.ndarray:
        top_fraction = self.cfg.region_support_top_fraction if top_fraction is None else top_fraction
        scores = np.asarray(region_scores, dtype=np.float32)
        pos = scores[scores > 0]
        out = np.zeros_like(scores, dtype=np.float32)
        if pos.size == 0:
            return out
        thr = np.quantile(pos, max(0.0, min(1.0, 1.0 - top_fraction)))
        out[scores >= thr] = 1.0
        return out

    def restrict_to_roi(self, score_map, roi_mask):
        score_map = np.asarray(score_map, dtype=np.float32).copy()
        if roi_mask is None:
            return score_map
        score_map[roi_mask == 0] = 0.0
        return score_map

    def mask_inside_roi(self, bin_mask, roi_mask):
        bin_mask = np.asarray(bin_mask, dtype=np.uint8)
        if roi_mask is None:
            return bin_mask
        return (bin_mask.astype(bool) & roi_mask.astype(bool)).astype(np.uint8)

    def region_mean(self, segments, value_map):
        segments = np.asarray(segments, dtype=np.int32)
        value_map = np.asarray(value_map, dtype=np.float32)
        valid = segments >= 0
        if not np.any(valid):
            return np.zeros(0, dtype=np.float32)
        n = int(segments[valid].max() + 1)
        sums = np.bincount(segments[valid].ravel(), weights=value_map[valid].ravel(), minlength=n)
        counts = np.bincount(segments[valid].ravel(), minlength=n)
        return (sums / np.maximum(counts, 1)).astype(np.float32)

    def region_scores_to_map(self, segments, region_scores):
        segments = np.asarray(segments, dtype=np.int32)
        region_scores = np.asarray(region_scores, dtype=np.float32)
        pix = np.zeros(segments.shape, dtype=np.float32)
        valid = segments >= 0
        if np.any(valid) and len(region_scores) > 0:
            pix[valid] = region_scores[segments[valid]]
        return pix

    def valid_region_ids(self, segments):
        ids = np.unique(np.asarray(segments, dtype=np.int32))
        return ids[ids >= 0]

    def n_regions(self, segments):
        ids = self.valid_region_ids(segments)
        if len(ids) == 0:
            return 0
        return int(ids.max() + 1)

    def selected_triplet_methods(self) -> Tuple[str, str, str]:
        return selected_triplet_from_cfg(self.cfg)

    def resolve_requested_methods(self, requested_methods: Optional[List[str]] = None) -> List[str]:
        if requested_methods is None:
            methods = list(ALL_BASELINE_METHODS)
        else:
            methods = [m for m in requested_methods if m in ALL_BASELINE_METHODS]

        if requested_methods is None or "Proposed_VisualMergeConsensus" in requested_methods:
            for m in self.selected_triplet_methods():
                if m not in methods:
                    methods.append(m)

        ordered = [m for m in ALL_BASELINE_METHODS if m in methods]
        return ordered

    def build_initial_segments(self, image_np, roi_mask):
        roi_mask = roi_mask.astype(bool)
        y0, y1, x0, x1 = self.mask_bbox(roi_mask, margin=0)
        img_crop = image_np[y0:y1, x0:x1].copy()
        roi_crop = roi_mask[y0:y1, x0:x1]

        if roi_crop.sum() == 0:
            return -np.ones(roi_mask.shape, dtype=np.int32)

        img_crop = img_crop.astype(np.float32) / 255.0
        fill_vec = img_crop[roi_crop].mean(axis=0)
        img_crop[~roi_crop] = fill_vec

        seg_crop = slic(
            img_crop,
            n_segments=self.cfg.initial_n_segments,
            compactness=self.cfg.initial_compactness,
            sigma=self.cfg.initial_sigma,
            start_label=0,
            convert2lab=False,
            channel_axis=-1,
        ).astype(np.int32)

        full_seg = -np.ones(roi_mask.shape, dtype=np.int32)
        full_seg[y0:y1, x0:x1] = seg_crop
        full_seg[~roi_mask] = -1
        full_seg[y0:y1, x0:x1][~roi_crop] = -1

        full_seg = self.split_disconnected_segments_keep_minus_one(full_seg)
        full_seg = self.relabel_segments_keep_minus_one(full_seg)

        valid_ids = self.valid_region_ids(full_seg)
        if len(valid_ids) == 0:
            full_seg = -np.ones_like(roi_mask, dtype=np.int32)
            full_seg[roi_mask > 0] = 0

        return full_seg

    def split_disconnected_segments_keep_minus_one(self, segments):
        segments = np.asarray(segments, dtype=np.int32)
        out = -np.ones_like(segments, dtype=np.int32)
        next_id = 0
        for sid in self.valid_region_ids(segments):
            comp, ncomp = ndi.label(segments == sid)
            for k in range(1, ncomp + 1):
                out[comp == k] = next_id
                next_id += 1
        return out

    def relabel_segments_keep_minus_one(self, seg):
        seg = np.asarray(seg, dtype=np.int32)
        out = -np.ones_like(seg, dtype=np.int32)
        uniq = [int(u) for u in np.unique(seg) if int(u) >= 0]
        mapping = {u: i for i, u in enumerate(uniq)}
        for u, i in mapping.items():
            out[seg == u] = i
        return out

    def build_adjacency(self, segments: np.ndarray) -> List[Tuple[int, int]]:
        seg = np.asarray(segments, dtype=np.int32)
        edges = set()

        a = seg[:, :-1]
        b = seg[:, 1:]
        mask = (a >= 0) & (b >= 0) & (a != b)
        for u, v in zip(a[mask].ravel(), b[mask].ravel()):
            if u > v:
                u, v = v, u
            edges.add((int(u), int(v)))

        a = seg[:-1, :]
        b = seg[1:, :]
        mask = (a >= 0) & (b >= 0) & (a != b)
        for u, v in zip(a[mask].ravel(), b[mask].ravel()):
            if u > v:
                u, v = v, u
            edges.add((int(u), int(v)))

        return sorted(list(edges))

    def compute_visual_features(self, image_np, segments):
        gray = self.to_gray_float(image_np)
        grad = ndi.gaussian_gradient_magnitude(gray, sigma=1.0)
        n_regions = self.n_regions(segments)

        mean_int = np.zeros(n_regions, dtype=np.float32)
        std_int = np.zeros(n_regions, dtype=np.float32)
        mean_grad = np.zeros(n_regions, dtype=np.float32)

        for rid in range(n_regions):
            mask = segments == rid
            vals = gray[mask]
            gvals = grad[mask]
            if vals.size == 0:
                continue
            mean_int[rid] = float(vals.mean())
            std_int[rid] = float(vals.std())
            mean_grad[rid] = float(gvals.mean())

        feats = np.stack(
            [normalize01(mean_int), normalize01(std_int), normalize01(mean_grad)],
            axis=1,
        ).astype(np.float32)
        return feats

    def similarity_between_regions(self, feat_i, feat_j):
        d = np.sqrt(
            self.cfg.merge_w_mean * float((feat_i[0] - feat_j[0]) ** 2) +
            self.cfg.merge_w_std * float((feat_i[1] - feat_j[1]) ** 2) +
            self.cfg.merge_w_grad * float((feat_i[2] - feat_j[2]) ** 2)
        )
        sim = np.exp(-(d ** 2) / (2.0 * (self.cfg.merge_similarity_sigma ** 2) + 1e-8))
        return float(sim)

    def merge_segments_by_visual_similarity(self, image_np, roi_mask, initial_segments):
        n_regions = self.n_regions(initial_segments)
        if n_regions <= 1:
            return initial_segments.copy(), []

        feats = self.compute_visual_features(image_np, initial_segments)
        edges = self.build_adjacency(initial_segments)
        uf = UnionFind(n_regions)
        component_sizes = np.ones(n_regions, dtype=np.int32)
        merge_rows = []

        scored_edges = []
        for u, v in edges:
            sim = self.similarity_between_regions(feats[u], feats[v])
            merge_rows.append({"region_u": u, "region_v": v, "similarity": sim})
            scored_edges.append((u, v, sim))

        scored_edges.sort(key=lambda x: x[2], reverse=True)

        for u, v, sim in scored_edges:
            if sim < self.cfg.merge_similarity_threshold:
                continue

            ru = uf.find(u)
            rv = uf.find(v)
            if ru == rv:
                continue

            if component_sizes[ru] + component_sizes[rv] > int(self.cfg.max_initial_regions_per_merged_zone):
                continue

            uf.union(ru, rv)
            new_root = uf.find(ru)
            if new_root == ru:
                component_sizes[ru] = component_sizes[ru] + component_sizes[rv]
            else:
                component_sizes[rv] = component_sizes[ru] + component_sizes[rv]

        mapping = {}
        next_id = 0
        merged = -np.ones_like(initial_segments, dtype=np.int32)
        for rid in range(n_regions):
            root = uf.find(rid)
            if root not in mapping:
                mapping[root] = next_id
                next_id += 1
            merged[initial_segments == rid] = mapping[root]

        merged[roi_mask == 0] = -1
        merged = self.relabel_segments_keep_minus_one(merged)
        return merged, merge_rows

    def compute_region_faithfulness(self, model, image_np_for_model, segments, target_label=1):
        n_regions = self.n_regions(segments)
        if n_regions == 0:
            return np.zeros(0, dtype=np.float32)

        base_prob = float(
            self.predict_proba_numpy(
                model=model,
                images_np=image_np_for_model[None, ...],
                batch_size=1,
            )[0, int(target_label)]
        )
        drops = np.zeros(n_regions, dtype=np.float32)
        fill = np.mean(
            image_np_for_model.reshape(-1, image_np_for_model.shape[-1]),
            axis=0,
        ).clip(0, 255).astype(np.uint8)

        for sid in range(n_regions):
            masked = image_np_for_model.astype(np.float32).copy()
            masked[segments == sid] = fill
            masked = np.clip(masked, 0, 255).astype(np.uint8)

            prob = float(
                self.predict_proba_numpy(
                    model=model,
                    images_np=masked[None, ...],
                    batch_size=1,
                )[0, int(target_label)]
            )
            drops[sid] = max(0.0, base_prob - prob)

        return normalize01(drops.astype(np.float32))

    def region_stability_from_runs(self, runs):
        runs = np.asarray(runs, dtype=np.float32)
        if runs.ndim == 1:
            runs = runs[None, :]
        if runs.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        if runs.shape[0] == 1:
            return np.ones(runs.shape[1], dtype=np.float32)

        norm_runs = np.stack([normalize01(np.clip(r, 0.0, None)) for r in runs], axis=0)
        std = np.std(norm_runs, axis=0)
        return (1.0 - normalize01(std)).astype(np.float32)

    def combine_region_scores(self, method_runs_region: Dict[str, np.ndarray], faithfulness_scores: np.ndarray):
        method_mean = {}
        method_stability = {}
        method_support = {}

        for name, runs in method_runs_region.items():
            runs = np.asarray(runs, dtype=np.float32)
            if runs.ndim == 1:
                runs = runs[None, :]
            runs = np.clip(runs, 0.0, None)

            if runs.shape[1] == 0:
                method_mean[name] = np.zeros(0, dtype=np.float32)
                method_stability[name] = np.zeros(0, dtype=np.float32)
                method_support[name] = np.zeros(0, dtype=np.float32)
            else:
                mean_scores = normalize01(np.mean(runs, axis=0))
                method_mean[name] = mean_scores
                method_stability[name] = self.region_stability_from_runs(runs)
                method_support[name] = self.region_top_fraction_support(
                    mean_scores,
                    self.cfg.region_support_top_fraction,
                )

        method_names = list(method_mean.keys())
        if len(method_names) == 0 or len(method_mean[method_names[0]]) == 0:
            return {
                "method_mean": method_mean,
                "method_stability": method_stability,
                "method_support": method_support,
                "agreement": np.zeros(0, dtype=np.float32),
                "stability": np.zeros(0, dtype=np.float32),
                "faithfulness": np.zeros(0, dtype=np.float32),
                "faithfulness_gate": np.zeros(0, dtype=np.float32),
                "support_count": np.zeros(0, dtype=np.int32),
                "agreement_gate": np.zeros(0, dtype=np.float32),
                "stability_gate": np.zeros(0, dtype=np.float32),
                "importance_gate": np.zeros(0, dtype=np.float32),
                "evidence": np.zeros(0, dtype=np.float32),
                "consensus": np.zeros(0, dtype=np.float32),
                "proposed_binary_zone_mask": np.zeros(0, dtype=np.float32),
            }

        method_matrix = np.stack([method_mean[k] for k in method_names], axis=0)
        support_matrix = np.stack([method_support[k] for k in method_names], axis=0)

        evidence = np.median(method_matrix, axis=0).astype(np.float32)
        support_count = np.sum(support_matrix > 0, axis=0).astype(np.int32)
        agreement = (support_count.astype(np.float32) / float(len(method_names))).astype(np.float32)
        stability = np.mean(
            np.stack([method_stability[k] for k in method_names], axis=0),
            axis=0,
        ).astype(np.float32)
        faithfulness = normalize01(np.asarray(faithfulness_scores, dtype=np.float32))

        faithfulness_gate = (
            faithfulness >= float(self.cfg.faithfulness_importance_threshold)
        ).astype(np.float32)
        agreement_gate = (
            support_count >= int(self.cfg.agreement_min_supported_methods)
        ).astype(np.float32)
        stability_gate = (
            stability >= float(self.cfg.stability_importance_threshold)
        ).astype(np.float32)

        importance_gate = (
            ((faithfulness_gate > 0) | (agreement_gate > 0)) & (stability_gate > 0)
        ).astype(np.float32)

        wsum = (
            self.cfg.weight_agreement +
            self.cfg.weight_stability +
            self.cfg.weight_faithfulness
        )
        quality = (
            self.cfg.weight_agreement * agreement +
            self.cfg.weight_stability * stability +
            self.cfg.weight_faithfulness * faithfulness
        ) / max(wsum, 1e-8)

        consensus = normalize01(evidence * quality * importance_gate).astype(np.float32)
        proposed_binary_zone_mask = importance_gate.astype(np.float32)

        return {
            "method_mean": method_mean,
            "method_stability": method_stability,
            "method_support": method_support,
            "agreement": agreement,
            "stability": stability,
            "faithfulness": faithfulness,
            "faithfulness_gate": faithfulness_gate,
            "support_count": support_count,
            "agreement_gate": agreement_gate,
            "stability_gate": stability_gate,
            "importance_gate": importance_gate,
            "evidence": evidence,
            "consensus": consensus,
            "proposed_binary_zone_mask": proposed_binary_zone_mask,
        }

    def segments_for_visualization(self, segments):
        vis = np.asarray(segments, dtype=np.int32).copy()
        if np.any(vis < 0):
            fill_id = int(vis[vis >= 0].max() + 1) if np.any(vis >= 0) else 0
            vis[vis < 0] = fill_id
        return vis