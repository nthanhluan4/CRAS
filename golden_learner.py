"""
golden_learner.py
PatchCore-style Few-Shot Memory Bank Self-Learning Module & Anti-Poisoning Safeguards
Enables industrial machinery to learn new fabric textures in seconds with production freeze.
"""

import hashlib
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.ndimage import gaussian_filter


PROFILES_DIR = "results/profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)

# predict() maps distance d -> d / (threshold * NORM_SCALE); the UI default decision level is 0.5
NORM_SCALE = 1.8
DEFAULT_DECISION_LEVEL = 0.5
CALIBRATION_MARGIN = 1.10
SEED = 0
# Tile-by-tile analysis of large frames (e.g. 512). Off by default: on ITDD defects mosaicked into 2048px
# frames it was ~5x slower (calibration ~35x) without better recall. Enable only if real small defects are missed.
DEFAULT_TILE_PX = None
MAX_GREEDY_POOL = 40000        # random pre-selection before greedy coreset, keeps calibration fast


def tile_boxes(width: int, height: int, tile_px: Optional[int]) -> List[Tuple[int, int, int, int]]:
    """
    Square tiles (x0, y0, x1, y1) covering the image; the last row/column is flushed to the border
    (slight overlap) so every tile keeps the same physical scale. Small images -> one whole-image tile.
    """
    if not tile_px or max(width, height) <= tile_px * 1.25:
        return [(0, 0, width, height)]
    t = min(tile_px, width, height)

    def starts(size):
        n = int(np.ceil(size / t))
        return sorted({min(i * t, size - t) for i in range(n)})

    return [(x, y, x + t, y + t) for y in starts(height) for x in starts(width)]


def _preselect(features: torch.Tensor, limit: int = MAX_GREEDY_POOL, seed: int = SEED) -> torch.Tensor:
    if features.shape[0] <= limit:
        return features
    idx = torch.randperm(features.shape[0], generator=torch.Generator().manual_seed(seed))[:limit]
    return features[idx]


def greedy_coreset(features: torch.Tensor, target_size: int, proj_dim: int = 128, seed: int = SEED) -> torch.Tensor:
    """
    PatchCore greedy k-center coreset selection (on a random projection for speed).
    Unlike random sampling, it keeps rare-but-normal texture patches, which otherwise
    become false positives in production.
    """
    n = features.shape[0]
    if n <= target_size:
        return torch.arange(n)
    gen = torch.Generator().manual_seed(seed)
    if features.shape[1] > proj_dim:
        proj = torch.randn(features.shape[1], proj_dim, generator=gen) / np.sqrt(proj_dim)
        reduced = features @ proj
    else:
        reduced = features
    selected = [int(torch.randint(n, (1,), generator=gen).item())]
    min_dists = torch.cdist(reduced, reduced[selected[-1]:selected[-1] + 1]).squeeze(1)
    for _ in range(target_size - 1):
        idx = int(torch.argmax(min_dists).item())
        selected.append(idx)
        min_dists = torch.minimum(min_dists, torch.cdist(reduced, reduced[idx:idx + 1]).squeeze(1))
    return torch.tensor(selected)


class GoldenMemoryLearner:
    """
    Industrial Few-Shot Self-Learning Engine based on PatchCore Coreset Memory Bank.
    Features strict anti-poisoning safeguards and production freezing.
    """

    def __init__(self, device: Optional[torch.device] = None, img_size: int = 288,
                 tile_px: Optional[int] = DEFAULT_TILE_PX):
        self.device = device if device is not None else torch.device("cpu")
        self.img_size = img_size
        self.tile_px = tile_px            # tiling used by the ACTIVE profile (taken from the profile on load)
        self.default_tile_px = tile_px    # tiling for newly calibrated profiles

        # Pretrained backbone (ResNet18 runs in ~35ms on CPU)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.to(self.device)
        self.backbone.eval()

        # Feature hooks for multi-scale local patches
        self.features = {}
        self._register_hooks()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Active Memory Bank State
        self.memory_bank: Optional[torch.Tensor] = None
        self.active_profile_name: str = "Uninitialized"
        self.is_locked: bool = True
        self.calibrated_threshold: float = 0.50
        self.metadata: Dict[str, Any] = {}

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook

        self.backbone.layer2.register_forward_hook(get_hook("layer2"))
        self.backbone.layer3.register_forward_hook(get_hook("layer3"))

    def extract_patch_features(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Extracts multi-scale local patch representations from layer2 and layer3 for the whole image.
        Returns tensor of shape [H*W, feature_dim].
        """
        return self._extract_batch([pil_image])[0]

    def _extract_batch(self, pil_images: List[Image.Image], batch_size: int = 16) -> List[torch.Tensor]:
        out = []
        for i in range(0, len(pil_images), batch_size):
            tensor = torch.stack([self.transform(im) for im in pil_images[i:i + batch_size]]).to(self.device)
            with torch.no_grad():
                _ = self.backbone(tensor)
                l2 = self.features["layer2"]  # [B, 128, H2, W2]
                l3 = self.features["layer3"]  # [B, 256, H3, W3]
                # Upsample layer3 to layer2 spatial resolution, 3x3 local patch aggregation
                l3_upsampled = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                feats = F.avg_pool2d(torch.cat([l2, l3_upsampled], dim=1), kernel_size=3, stride=1, padding=1)
                feats = feats.permute(0, 2, 3, 1).reshape(feats.shape[0], -1, feats.shape[1])  # [B, N, 384]
                feats = F.normalize(feats, p=2, dim=2)
            out.extend(f.cpu() for f in feats)
        return out

    def extract_tiled_features(self, pil_image: Image.Image, tile_px: Optional[int]):
        """Per-tile patch features ([N, 384] each) and the tile boxes in original-image pixels."""
        boxes = tile_boxes(pil_image.size[0], pil_image.size[1], tile_px)
        crops = [pil_image.crop(b) for b in boxes]
        return self._extract_batch(crops), boxes

    def calibrate(
        self,
        golden_images: List[Image.Image],
        profile_name: str = "Fabric_SKU_Default",
        target_coreset_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Self-learns a new fabric distribution in seconds from 3-10 golden samples.
        Includes outlier rejection to prevent learning contaminated/defective fabric.
        Large camera frames are learnt tile by tile (same tiling as used by predict()).
        """
        t0 = time.time()
        if not golden_images:
            raise ValueError("Cần ít nhất 1 ảnh Mẫu Vàng (Golden Sample) để học!")

        tile_px = self.default_tile_px
        all_patch_batches = []      # per image: [T*N, D]
        all_tile_batches = []       # per image: list of per-tile [N, D]
        for img in golden_images:
            tiles, _ = self.extract_tiled_features(img, tile_px)
            all_tile_batches.append(tiles)
            all_patch_batches.append(torch.cat(tiles, dim=0))

        tiles_per_image = max(len(t) for t in all_tile_batches)
        if target_coreset_size is None:
            target_coreset_size = 1200 if tiles_per_image == 1 else 3000

        # Anti-Poisoning Outlier Guard: Check mutual patch-level consistency
        outlier_warnings = []
        if len(golden_images) >= 3:
            for idx, patches in enumerate(all_patch_batches):
                other_batches = [p for j, p in enumerate(all_patch_batches) if j != idx]
                ref_sub = torch.cat(other_batches, dim=0)
                if ref_sub.shape[0] > 1000:
                    ref_sub = ref_sub[torch.randperm(ref_sub.shape[0], generator=torch.Generator().manual_seed(SEED))[:1000]]
                dists = torch.cdist(patches, ref_sub).min(dim=1)[0]
                p98_dist = float(torch.quantile(dists, 0.98).item())
                if p98_dist > 0.58:
                    outlier_warnings.append(
                        f"⚠️ CẢNH BÁO MẪU BẨN: Ảnh mẫu #{idx+1} có đặc trưng dị biệt cao ({p98_dist:.3f} > ngưỡng an toàn 0.580). "
                        "Ảnh này có nguy cơ chứa nếp gấp, đốm ố hoặc sợi lỗi!"
                    )

        # Merge all patches
        combined_patches = torch.cat(all_patch_batches, dim=0)

        # Greedy k-center coreset (PatchCore) on a random projection
        with torch.no_grad():
            pool = _preselect(combined_patches)
            coreset_bank = pool[greedy_coreset(pool, target_coreset_size)]
            calibrated_thresh, calibration_method = self._calibrate_threshold(all_tile_batches, coreset_bank, target_coreset_size)

        elapsed = time.time() - t0

        # Update active state and lock for production
        self.memory_bank = coreset_bank.cpu()
        self.active_profile_name = profile_name
        self.is_locked = True
        self.calibrated_threshold = calibrated_thresh
        self.tile_px = tile_px

        # Create cryptographic hash for verification
        bank_bytes = self.memory_bank.numpy().tobytes()
        profile_hash = hashlib.sha256(bank_bytes).hexdigest()[:12]

        self.metadata = {
            "profile_name": profile_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_golden_samples": len(golden_images),
            "bank_vectors": self.memory_bank.shape[0],
            "feature_dim": self.memory_bank.shape[1],
            "calibrated_threshold": calibrated_thresh,
            "calibration_method": calibration_method,
            "tile_px": tile_px,
            "tiles_per_image": tiles_per_image,
            "profile_hash": profile_hash,
            "learning_time_sec": round(elapsed, 2),
            "is_locked": True,
            "outlier_warnings": outlier_warnings,
        }

        # Auto-save profile
        self.save_profile(profile_name)

        return self.metadata

    def _smoothed_distance_grid(self, patches: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
        nn_dists = torch.cdist(patches.cpu(), bank.cpu()).min(dim=1)[0]
        grid_dim = int(np.sqrt(patches.shape[0]))
        return gaussian_filter(nn_dists.reshape(grid_dim, grid_dim).numpy(), sigma=1.5)

    def _calibrate_threshold(self, tile_batches: List[List[torch.Tensor]], full_bank: torch.Tensor, coreset_size: int) -> Tuple[float, str]:
        """
        Leave-one-out calibration: each golden image is scored against a bank built from the OTHER
        images, exactly like an unseen production frame. The threshold is set so that the default
        decision level (0.5) sits CALIBRATION_MARGIN above the worst (p99.9) held-out normal patch.
        """
        n_img = len(tile_batches)
        if n_img >= 2:
            held_out = []
            per_fold = max(coreset_size // n_img * (n_img - 1), 1)
            for i, tiles in enumerate(tile_batches):
                others = _preselect(torch.cat([torch.cat(t, dim=0) for j, t in enumerate(tile_batches) if j != i], dim=0))
                fold_bank = others[greedy_coreset(others, per_fold)]
                for tile_patches in tiles:
                    held_out.append(self._smoothed_distance_grid(tile_patches, fold_bank).ravel())
            q = float(np.percentile(np.concatenate(held_out), 99.9))
            method = f"leave-one-out ({n_img} ảnh)"
        else:
            # Single sample: fall back to intra-bank nearest-neighbour spread (less reliable)
            dists = torch.cdist(full_bank, full_bank)
            dists.fill_diagonal_(float("inf"))
            q = float(torch.quantile(dists.min(dim=1)[0], 0.99).item()) * 1.5
            method = "self-distance (1 ảnh - kém tin cậy, nên dùng >= 3 ảnh)"
        thresh = CALIBRATION_MARGIN * q / (NORM_SCALE * DEFAULT_DECISION_LEVEL)
        return round(thresh, 4), method

    def predict(self, pil_image: Image.Image) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates patch-level anomaly distance to the locked Golden Memory Bank.
        Returns:
            anomaly_score: float (0.0 to 1.0)
            heatmap_resized: np.ndarray (RGB heatmap)
            overlay: np.ndarray (Overlay on raw image)
            norm_resized: np.ndarray (0..1 anomaly map, 0.5 = calibrated decision level)
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory Bank chưa được hiệu chuẩn! Vui lòng học Mẫu Vàng trước.")

        orig_w, orig_h = pil_image.size
        tiles, boxes = self.extract_tiled_features(pil_image, self.tile_px)
        thresh = max(self.calibrated_threshold, 1e-4)

        # Nearest neighbor patch distance per tile, smoothed on the tile grid, normalized to the calibrated
        # threshold and pasted back (max over the small overlaps of border tiles)
        norm_resized = np.zeros((orig_h, orig_w), np.float32)
        with torch.no_grad():
            for patches, (x0, y0, x1, y1) in zip(tiles, boxes):
                smoothed = self._smoothed_distance_grid(patches, self.memory_bank)
                tile_map = np.clip(smoothed / (thresh * NORM_SCALE), 0.0, 1.0).astype(np.float32)
                tile_map = cv2.resize(tile_map, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                np.maximum(norm_resized[y0:y1, x0:x1], tile_map, out=norm_resized[y0:y1, x0:x1])

        # Image-level anomaly score (top 99.5th percentile)
        score = float(np.percentile(norm_resized, 99.5))

        # Color heatmap
        heatmap_resized = cv2.cvtColor(cv2.applyColorMap((norm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        raw_rgb = np.array(pil_image)
        overlay = cv2.addWeighted(raw_rgb, 0.60, heatmap_resized, 0.40, 0)
        return score, heatmap_resized, overlay, norm_resized

    def save_profile(self, profile_name: str) -> str:
        """
        Saves the immutable memory bank profile to disk.
        """
        clean_name = "".join(c for c in profile_name if c.isalnum() or c in ("_", "-"))
        filepath = os.path.join(PROFILES_DIR, f"{clean_name}.bank")
        save_dict = {
            "metadata": self.metadata,
            "memory_bank": self.memory_bank,
            "threshold": self.calibrated_threshold,
            "is_locked": True,
        }
        torch.save(save_dict, filepath)
        return filepath

    def load_profile(self, filepath: str) -> Dict[str, Any]:
        """
        Loads an immutable memory bank profile from disk and freezes it.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file Profile: {filepath}")

        data = torch.load(filepath, map_location="cpu")
        self.memory_bank = data["memory_bank"].cpu()
        self.metadata = data.get("metadata", {})
        self.calibrated_threshold = data.get("threshold", 0.5)
        self.active_profile_name = self.metadata.get("profile_name", os.path.basename(filepath))
        # Profiles made before tiling existed were learnt on the whole (resized) image
        self.tile_px = self.metadata.get("tile_px", None)
        self.is_locked = True
        return self.metadata

    @staticmethod
    def list_available_profiles() -> List[Dict[str, str]]:
        """
        Lists all available Golden Sample profiles saved on disk.
        """
        profiles = []
        if not os.path.exists(PROFILES_DIR):
            return profiles

        for f in sorted(os.listdir(PROFILES_DIR)):
            if f.endswith(".bank"):
                p_path = os.path.join(PROFILES_DIR, f)
                file_info = {
                    "size_kb": round(os.path.getsize(p_path) / 1024.0, 1),
                    "modified_time": datetime.fromtimestamp(os.path.getmtime(p_path)).strftime("%Y-%m-%d %H:%M:%S"),
                }
                try:
                    data = torch.load(p_path, map_location="cpu")
                    meta = data.get("metadata", {})
                    name = meta.get("profile_name", f.replace(".bank", ""))
                    date = meta.get("created_at", "N/A")
                    profiles.append({
                        "filename": f,
                        "path": p_path,
                        "display_name": f"🔒 {name} ({date})",
                        "profile_name": name,
                        **file_info,
                    })
                except Exception:
                    profiles.append({
                        "filename": f,
                        "path": p_path,
                        "display_name": f"🔒 {f}",
                        "profile_name": f,
                        **file_info,
                    })
        return profiles
