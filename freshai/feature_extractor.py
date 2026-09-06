"""
FreshAI - Stage 4: Color & Texture Feature Extraction
Tool: OpenCV + NumPy

Objective:
Extract measurable numerical color and texture features from the detected produce crop:
• Color Analysis: RGB, HSV, CIELAB color spaces + Percentage of color regions (green, yellow, red, brown, dark, mold).
• Texture Analysis: GLCM features (Contrast, Dissimilarity, Homogeneity, Energy, Correlation) + Surface roughness.

Processing Flow:
Detected / Cropped Produce
 │
 ▼
 Image Preprocessing
 │
 ┌─────┴──────┐
 ▼             ▼
Color Analysis Texture Analysis
 │             │
 ▼             ▼
RGB / HSV / LAB GLCM Features
 │             │
 └─────┬──────┘
       ▼
 Numerical Features
       │
       ▼
 Feature Vector
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ColorFeatures:
    # RGB Space
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    std_r: float = 0.0
    std_g: float = 0.0
    std_b: float = 0.0
    r_g_ratio: float = 1.0
    r_b_ratio: float = 1.0

    # HSV Space
    avg_hue: float = 0.0           # 0.0 to 180.0 (or degrees 0 - 360)
    avg_saturation: float = 0.0    # 0.0 to 100.0%
    avg_brightness: float = 0.0    # 0.0 to 100.0%
    std_hue: float = 0.0
    std_saturation: float = 0.0
    hue_entropy: float = 0.0

    # CIELAB Space
    mean_l: float = 0.0            # Lightness (0 to 100)
    mean_a: float = 0.0            # -128 (green) to +127 (red)
    mean_b_lab: float = 0.0        # -128 (blue) to +127 (yellow)
    chroma: float = 0.0            # sqrt(a^2 + b^2)

    # Color Region Percentages
    green_region_pct: float = 0.0  # Chlorophyll / unripe / fresh leaf
    yellow_region_pct: float = 0.0 # Carotenoids / ripening stage
    red_region_pct: float = 0.0    # Anthocyanins / ripe red
    brown_region_pct: float = 0.0  # Enzymatic browning / aging
    dark_decay_pct: float = 0.0    # Necrotic decay / black spots
    pale_mold_pct: float = 0.0     # Pale fungal mycelium

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_rgb": [round(self.mean_r, 1), round(self.mean_g, 1), round(self.mean_b, 1)],
            "std_rgb": [round(self.std_r, 1), round(self.std_g, 1), round(self.std_b, 1)],
            "average_hue": round(self.avg_hue, 1),
            "saturation": round(self.avg_saturation, 1),
            "brightness": round(self.avg_brightness, 1),
            "cielab": {"L": round(self.mean_l, 1), "a": round(self.mean_a, 1), "b": round(self.mean_b_lab, 1), "chroma": round(self.chroma, 1)},
            "color_regions": {
                "green_unripe_pct": round(self.green_region_pct, 1),
                "yellow_ripe_pct": round(self.yellow_region_pct, 1),
                "red_ripe_pct": round(self.red_region_pct, 1),
                "brown_decay_pct": round(self.brown_region_pct, 1),
                "dark_spot_pct": round(self.dark_decay_pct, 1),
                "pale_mold_pct": round(self.pale_mold_pct, 1),
            }
        }

    def to_vector(self) -> List[float]:
        return [
            self.mean_r, self.mean_g, self.mean_b,
            self.std_r, self.std_g, self.std_b,
            self.r_g_ratio, self.r_b_ratio,
            self.avg_hue, self.avg_saturation, self.avg_brightness,
            self.std_hue, self.std_saturation, self.hue_entropy,
            self.mean_l, self.mean_a, self.mean_b_lab, self.chroma,
            self.green_region_pct, self.yellow_region_pct, self.red_region_pct,
            self.brown_region_pct, self.dark_decay_pct, self.pale_mold_pct,
        ]


@dataclass
class TextureFeatures:
    contrast: float = 0.0          # GLCM Contrast (intensity variation)
    dissimilarity: float = 0.0     # GLCM Dissimilarity
    homogeneity: float = 0.0       # GLCM Homogeneity (smoothness)
    energy: float = 0.0            # GLCM Energy / Uniformity (ASM)
    correlation: float = 0.0       # GLCM Linear correlation
    entropy: float = 0.0           # GLCM Texture entropy
    roughness: float = 0.0         # Laplacian variance (wrinkling / shriveling)
    edge_density: float = 0.0      # Canny edge density (cracks / micro-fissures)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texture_contrast": round(self.contrast, 4),
            "texture_dissimilarity": round(self.dissimilarity, 4),
            "texture_homogeneity": round(self.homogeneity, 4),
            "texture_energy": round(self.energy, 4),
            "texture_correlation": round(self.correlation, 4),
            "texture_entropy": round(self.entropy, 4),
            "surface_roughness": round(self.roughness, 2),
            "edge_density": round(self.edge_density, 4),
        }

    def to_vector(self) -> List[float]:
        return [
            self.contrast, self.dissimilarity, self.homogeneity,
            self.energy, self.correlation, self.entropy,
            self.roughness, self.edge_density,
        ]


@dataclass
class ProduceExtractedFeatures:
    color: ColorFeatures = field(default_factory=ColorFeatures)
    texture: TextureFeatures = field(default_factory=TextureFeatures)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color": self.color.to_dict(),
            "texture": self.texture.to_dict(),
        }

    def to_vector(self) -> List[float]:
        return self.color.to_vector() + self.texture.to_vector()


class ProduceFeatureExtractor:
    """
    High-performance OpenCV + NumPy Feature Extractor.
    Calculates comprehensive color and GLCM texture descriptors from produce crops.
    """

    def __init__(self, glcm_levels: int = 16):
        """
        Args:
            glcm_levels: Number of quantized gray levels for GLCM (16 or 32 recommended for speed & stability).
        """
        self.glcm_levels = glcm_levels

    def extract(
        self,
        crop: Union[np.ndarray, Image.Image, str],
        produce_mask: Optional[np.ndarray] = None,
    ) -> ProduceExtractedFeatures:
        """
        Extract numerical color and texture features from a produce image or crop.
        """
        # Standardize to RGB numpy array
        if isinstance(crop, str):
            crop_np = np.array(Image.open(crop).convert("RGB"))
        elif isinstance(crop, Image.Image):
            crop_np = np.array(crop.convert("RGB"))
        elif isinstance(crop, np.ndarray):
            crop_np = crop.copy()
            if crop_np.ndim == 2:
                crop_np = cv2.cvtColor(crop_np, cv2.COLOR_GRAY2RGB)
            elif crop_np.shape[2] == 4:
                crop_np = cv2.cvtColor(crop_np, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(crop)}")

        h, w = crop_np.shape[:2]
        if h < 4 or w < 4:
            return ProduceExtractedFeatures()

        # Foreground mask
        if produce_mask is None:
            fg_mask = self._get_foreground_mask(crop_np)
        else:
            fg_mask = (produce_mask > 0).astype(np.uint8)

        # 1. Color features
        color_feats = self._extract_color_features(crop_np, fg_mask)

        # 2. Texture features (GLCM)
        texture_feats = self._extract_texture_features(crop_np, fg_mask)

        return ProduceExtractedFeatures(color=color_feats, texture=texture_feats)

    def _get_foreground_mask(self, crop_np: np.ndarray) -> np.ndarray:
        """Segment produce foreground from plain or backdrop surface."""
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]

        # Ignore pure white background and extreme black frame borders
        white_bg = (gray > 240) & (sat < 30)
        black_bg = (gray < 15)
        fg = (~(white_bg | black_bg)).astype(np.uint8)

        if np.sum(fg) < (crop_np.shape[0] * crop_np.shape[1] * 0.10):
            fg = np.ones((crop_np.shape[0], crop_np.shape[1]), dtype=np.uint8)
        return fg

    def _extract_color_features(self, crop_np: np.ndarray, fg_mask: np.ndarray) -> ColorFeatures:
        fg_indices = np.where(fg_mask > 0)
        if len(fg_indices[0]) == 0:
            fg_indices = (np.arange(crop_np.shape[0]), np.arange(crop_np.shape[1]))

        # RGB stats
        r = crop_np[:, :, 0][fg_indices].astype(np.float32)
        g = crop_np[:, :, 1][fg_indices].astype(np.float32)
        b = crop_np[:, :, 2][fg_indices].astype(np.float32)

        mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        std_r, std_g, std_b = float(np.std(r)), float(np.std(g)), float(np.std(b))
        r_g_ratio = mean_r / max(1.0, mean_g)
        r_b_ratio = mean_r / max(1.0, mean_b)

        # HSV stats
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        H = hsv[:, :, 0][fg_indices].astype(np.float32)  # 0 to 179
        S = hsv[:, :, 1][fg_indices].astype(np.float32)  # 0 to 255
        V = hsv[:, :, 2][fg_indices].astype(np.float32)  # 0 to 255

        avg_hue = float(np.mean(H))
        avg_sat = float(np.mean(S)) / 255.0 * 100.0  # 0 to 100%
        avg_val = float(np.mean(V)) / 255.0 * 100.0  # 0 to 100%
        std_hue = float(np.std(H))
        std_sat = float(np.std(S)) / 255.0 * 100.0

        # Hue distribution entropy
        hist, _ = np.histogram(H, bins=18, range=(0, 180), density=True)
        hist = hist[hist > 0]
        hue_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

        # CIELAB stats
        lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0][fg_indices].astype(np.float32) / 2.55   # 0 to 100
        A = lab[:, :, 1][fg_indices].astype(np.float32) - 128.0  # -128 to 127
        B = lab[:, :, 2][fg_indices].astype(np.float32) - 128.0  # -128 to 127

        mean_l = float(np.mean(L))
        mean_a = float(np.mean(A))
        mean_b_lab = float(np.mean(B))
        chroma = float(np.sqrt(mean_a**2 + mean_b_lab**2))

        # Color Region Percentages
        total_pixels = max(1, len(fg_indices[0]))
        # Green (Hue roughly 35 - 85 in OpenCV)
        green_mask = (H >= 35) & (H <= 85) & (S > 40)
        # Yellow (Hue roughly 20 - 34)
        yellow_mask = (H >= 20) & (H < 35) & (S > 50)
        # Red (Hue < 15 or Hue >= 165)
        red_mask = ((H < 18) | (H >= 165)) & (S > 50) & (V > 50)
        # Brown / Decay (Hue 10 - 25, low-mid value and saturation)
        brown_mask = (H >= 8) & (H <= 25) & (S >= 40) & (S <= 160) & (V < 110)
        # Dark decay / necrotic black spots
        dark_mask = (V < 45)
        # Pale / White mold
        mold_mask = (S < 35) & (V > 150) & (L > 65)

        green_pct = float(np.sum(green_mask)) / total_pixels * 100.0
        yellow_pct = float(np.sum(yellow_mask)) / total_pixels * 100.0
        red_pct = float(np.sum(red_mask)) / total_pixels * 100.0
        brown_pct = float(np.sum(brown_mask)) / total_pixels * 100.0
        dark_pct = float(np.sum(dark_mask)) / total_pixels * 100.0
        mold_pct = float(np.sum(mold_mask)) / total_pixels * 100.0

        return ColorFeatures(
            mean_r=mean_r,
            mean_g=mean_g,
            mean_b=mean_b,
            std_r=std_r,
            std_g=std_g,
            std_b=std_b,
            r_g_ratio=r_g_ratio,
            r_b_ratio=r_b_ratio,
            avg_hue=avg_hue,
            avg_saturation=avg_sat,
            avg_brightness=avg_val,
            std_hue=std_hue,
            std_saturation=std_sat,
            hue_entropy=hue_entropy,
            mean_l=mean_l,
            mean_a=mean_a,
            mean_b_lab=mean_b_lab,
            chroma=chroma,
            green_region_pct=green_pct,
            yellow_region_pct=yellow_pct,
            red_region_pct=red_pct,
            brown_region_pct=brown_pct,
            dark_decay_pct=dark_pct,
            pale_mold_pct=mold_pct,
        )

    def _extract_texture_features(self, crop_np: np.ndarray, fg_mask: np.ndarray) -> TextureFeatures:
        """
        Compute multi-scale Gray-Level Co-occurrence Matrix (GLCM) and surface dynamics.
        Isolates true produce skin / cuticle texture by eroding outer boundary background artifacts.
        """
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Erode foreground mask slightly to prevent bounding box edge transitions from contaminating texture
        if h > 10 and w > 10 and np.sum(fg_mask > 0) > 40:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inner_fg = cv2.erode(fg_mask, kernel, iterations=1)
            if np.sum(inner_fg > 0) < 20:
                inner_fg = fg_mask
        else:
            inner_fg = fg_mask

        # Surface roughness via Laplacian variance strictly on produce skin
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_fg = lap[inner_fg > 0]
        if len(lap_fg) > 10:
            roughness = float(np.var(lap_fg))
        else:
            roughness = float(np.var(lap))

        # Edge density via Canny on produce surface
        edges = cv2.Canny(gray, 50, 150)
        fg_count = max(1, int(np.sum(inner_fg > 0)))
        edge_density = float(np.sum((edges > 0) & (inner_fg > 0))) / fg_count

        # Adaptive contrast equalization to bring out subtle peel wrinkles & fungal mycelium
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq_gray = clahe.apply(gray)

        # Quantize gray image to glcm_levels (e.g. 16 levels: 0..15)
        quantized = (eq_gray // (256 // self.glcm_levels)).astype(np.int32)
        quantized = np.clip(quantized, 0, self.glcm_levels - 1)

        # Multi-scale GLCM calculation across both fine micro-scale (d=1) and meso-scale (d=2)
        # Directions: 0°, 90°, 135°, 45°
        levels = self.glcm_levels
        glcm = np.zeros((levels, levels), dtype=np.float32)

        directions = [
            # Micro-scale (d=1)
            (0, 1),   (1, 0),   (1, 1),   (1, -1),
            # Meso-scale (d=2) - reveals peel wrinkling and shriveling
            (0, 2),   (2, 0),   (2, 2),   (2, -2),
        ]

        for dy, dx in directions:
            if dy >= 0 and dx >= 0:
                p1 = quantized[:h - max(dy, 0), :w - max(dx, 0)]
                p2 = quantized[max(dy, 0):, max(dx, 0):]
                m1 = inner_fg[:h - max(dy, 0), :w - max(dx, 0)]
                m2 = inner_fg[max(dy, 0):, max(dx, 0):]
            elif dy >= 0 and dx < 0:
                p1 = quantized[:h - dy, -dx:]
                p2 = quantized[dy:, :w + dx]
                m1 = inner_fg[:h - dy, -dx:]
                m2 = inner_fg[dy:, :w + dx]
            else:
                continue

            valid = (m1 > 0) & (m2 > 0)
            if np.sum(valid) > 0:
                idx1 = p1[valid].ravel()
                idx2 = p2[valid].ravel()
                # Accumulate co-occurrences symmetrically
                np.add.at(glcm, (idx1, idx2), 1.0)
                np.add.at(glcm, (idx2, idx1), 1.0)

        # Normalize GLCM to joint probability distribution
        total_pairs = np.sum(glcm)
        if total_pairs > 0:
            P = glcm / total_pairs
        else:
            P = np.eye(levels, dtype=np.float32) / float(levels)

        # Grid indices
        i_grid, j_grid = np.indices((levels, levels))
        diff_sq = (i_grid - j_grid) ** 2
        diff_abs = np.abs(i_grid - j_grid)

        # 1. Contrast: sum(|i - j|^2 * P(i,j))
        contrast = float(np.sum(diff_sq * P))

        # 2. Dissimilarity: sum(|i - j| * P(i,j))
        dissimilarity = float(np.sum(diff_abs * P))

        # 3. Homogeneity: sum(P(i,j) / (1 + |i - j|^2))
        homogeneity = float(np.sum(P / (1.0 + diff_sq)))

        # 4. Energy (Uniformity / Angular Second Moment ASM): sum(P(i,j)^2)
        energy = float(np.sum(P ** 2))

        # 5. Correlation: sum((i - mu_i)(j - mu_j) P(i,j)) / (sigma_i * sigma_j)
        mu_i = np.sum(i_grid * P)
        mu_j = np.sum(j_grid * P)
        sigma_i = np.sqrt(np.sum(((i_grid - mu_i) ** 2) * P))
        sigma_j = np.sqrt(np.sum(((j_grid - mu_j) ** 2) * P))
        if sigma_i * sigma_j > 1e-7:
            correlation = float(np.sum((i_grid - mu_i) * (j_grid - mu_j) * P) / (sigma_i * sigma_j))
        else:
            correlation = 1.0

        # 6. Entropy: -sum(P * log2(P))
        P_nonzero = P[P > 1e-10]
        entropy = float(-np.sum(P_nonzero * np.log2(P_nonzero)))

        return TextureFeatures(
            contrast=contrast,
            dissimilarity=dissimilarity,
            homogeneity=homogeneity,
            energy=energy,
            correlation=correlation,
            entropy=entropy,
            roughness=roughness,
            edge_density=edge_density,
        )
