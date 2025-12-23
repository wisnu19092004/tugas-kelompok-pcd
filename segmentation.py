# segmentation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import io


# -----------------------------
# Helpers
# -----------------------------
def bgr_from_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar. Pastikan file gambar valid (jpg/png).")
    return img


def resize_keep_aspect(img_bgr: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr
    scale = max_side / float(m)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_mask(img_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    color = np.zeros_like(img_bgr)
    color[:, :, 1] = 255
    blended = img_bgr.copy()
    blended[mask01 == 1] = cv2.addWeighted(
        img_bgr[mask01 == 1], 1 - alpha, color[mask01 == 1], alpha, 0
    )
    return blended


def ensure_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# -----------------------------
# Metrics (requires GT mask)
# -----------------------------
def _as_binary01(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


def gt_mask_from_bytes(mask_bytes: bytes, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Decode GT mask image bytes and resize to match predicted mask size.
    Returns binary mask01 (H,W) uint8 in {0,1}.
    """
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Gagal membaca GT mask. Pastikan file mask valid (png/jpg).")
    h, w = target_hw
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return _as_binary01(m)


def iou_score(pred01: np.ndarray, gt01: np.ndarray) -> float:
    p = _as_binary01(pred01)
    g = _as_binary01(gt01)
    inter = int(np.sum(p & g))
    union = int(np.sum((p | g) > 0))
    return float(inter / (union + 1e-8))


def dice_score(pred01: np.ndarray, gt01: np.ndarray) -> float:
    p = _as_binary01(pred01)
    g = _as_binary01(gt01)
    inter = int(np.sum(p & g))
    denom = int(np.sum(p) + np.sum(g))
    return float((2.0 * inter) / (denom + 1e-8))


# -----------------------------
# Classic segmentation methods
# -----------------------------
def seg_otsu(img_bgr: np.ndarray, blur_ksize: int = 5, invert: bool = False) -> np.ndarray:
    gray = ensure_gray(img_bgr)
    k = max(1, int(blur_ksize))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = cv2.bitwise_not(th)
    return (th > 0).astype(np.uint8)


def seg_kmeans(img_bgr: np.ndarray, k: int = 3, attempts: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    K = int(max(2, k))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _ret, labels, centers = cv2.kmeans(
        Z, K, None, criteria, int(max(1, attempts)), cv2.KMEANS_PP_CENTERS
    )
    centers_u8 = centers.astype(np.uint8)
    labels2d = labels.reshape(img_bgr.shape[:2])
    return labels2d, centers_u8


def seg_watershed(
    img_bgr: np.ndarray,
    blur_ksize: int = 5,
    dist_thresh: float = 0.4,
    morph_ksize: int = 3
) -> np.ndarray:
    img = img_bgr.copy()
    gray = ensure_gray(img)

    k = max(1, int(blur_ksize))
    if k % 2 == 0:
        k += 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mk = max(1, int(morph_ksize))
    if mk % 2 == 0:
        mk += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))

    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dt = float(np.max(dist)) * float(np.clip(dist_thresh, 0.05, 0.95))
    _, sure_fg = cv2.threshold(dist, dt, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _nlabels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    return (markers > 1).astype(np.uint8)


# -----------------------------
# Deep segmentation (torchvision)
# -----------------------------
@dataclass
class DLResult:
    mask01: np.ndarray
    class_map: np.ndarray
    num_classes: int


def _lazy_import_torchvision():
    try:
        import torch  # noqa
        from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50  # noqa
        return True
    except Exception:
        return False


def seg_torchvision(
    img_bgr: np.ndarray,
    model_name: str = "deeplabv3_resnet50",
    target_class: int = 15,
    conf_thresh: float = 0.5,
    device: str = "cpu"
) -> DLResult:
    ok = _lazy_import_torchvision()
    if not ok:
        raise RuntimeError("PyTorch/torchvision belum terpasang. Install: pip install torch torchvision")

    import torch
    from torchvision import transforms
    from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

    model_name = model_name.strip().lower()
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT")
    elif model_name == "fcn_resnet50":
        model = fcn_resnet50(weights="DEFAULT")
    else:
        raise ValueError("model_name tidak dikenal. Pilih: deeplabv3_resnet50 / fcn_resnet50")

    model.eval()
    dev = torch.device(device if device in ["cpu", "cuda"] else "cpu")
    model.to(dev)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(img_rgb).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(x)["out"][0]  # (C,H,W)
        probs = torch.softmax(out, dim=0)
        class_map = torch.argmax(probs, dim=0)
        target_prob = probs[int(target_class)]
        mask01 = (target_prob >= float(conf_thresh)).to(torch.uint8)

    return DLResult(
        mask01=mask01.cpu().numpy().astype(np.uint8),
        class_map=class_map.cpu().numpy().astype(np.int32),
        num_classes=int(out.shape[0]),
    )


# -----------------------------
# UNet / FPN via segmentation-models-pytorch (SMP)
# -----------------------------
def _lazy_import_smp():
    try:
        import torch  # noqa
        import segmentation_models_pytorch as smp  # noqa
        return True
    except Exception:
        return False


def _smp_preprocess_rgb_float(img_rgb: np.ndarray, encoder_name: str):
    import torch
    import segmentation_models_pytorch as smp

    params = smp.encoders.get_preprocessing_params(encoder_name)
    mean = np.array(params["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(params["std"], dtype=np.float32).reshape(1, 1, 3)

    x = img_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0)  # 1CHW
    return x


def _extract_state_dict(obj: object) -> Dict[str, Any]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        return obj
    raise RuntimeError("Format weights tidak dikenali. Pastikan file berisi state_dict PyTorch.")


def seg_smp_binary(
    img_bgr: np.ndarray,
    arch: str = "unet",              # "unet" / "fpn"
    encoder_name: str = "resnet34",
    weights_bytes: Optional[bytes] = None,
    device: str = "cpu",
    conf_thresh: float = 0.5,
) -> np.ndarray:
    ok = _lazy_import_smp()
    if not ok:
        raise RuntimeError("Dependency untuk UNet/FPN belum lengkap. Install: pip install segmentation-models-pytorch timm")
    if not weights_bytes:
        raise RuntimeError("UNet/FPN memerlukan file weights (.pth/.pt). Silakan upload di UI.")

    import torch
    import segmentation_models_pytorch as smp

    arch = arch.strip().lower()
    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif arch == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError("arch tidak dikenali. Pilih: unet / fpn")

    # ✅ FIX: load from seekable buffer
    buf = io.BytesIO(weights_bytes)
    buf.seek(0)
    state = torch.load(buf, map_location="cpu")

    state_dict = _extract_state_dict(state)

    cleaned: Dict[str, Any] = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if isinstance(k, str) else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)

    model.eval()
    dev = torch.device(device if device in ["cpu", "cuda"] else "cpu")
    model.to(dev)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = _smp_preprocess_rgb_float(img_rgb, encoder_name).to(dev)

    with torch.no_grad():
        logits = model(x)  # (1,1,H,W)
        prob = torch.sigmoid(logits)[0, 0]  # (H,W)
        mask01 = (prob >= float(conf_thresh)).to(torch.uint8).cpu().numpy().astype(np.uint8)

    return mask01


# -----------------------------
# Unified API
# -----------------------------
def run_segmentation(
    img_bgr: np.ndarray,
    algo: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = params or {}
    algo_norm = algo.strip().lower()

    if algo_norm == "otsu":
        mask01 = seg_otsu(
            img_bgr,
            blur_ksize=int(params.get("blur_ksize", 5)),
            invert=bool(params.get("invert", False)),
        )
        overlay = overlay_mask(img_bgr, mask01, alpha=float(params.get("alpha", 0.5)))
        return {"mask01": mask01, "overlay": overlay, "meta": {"algo": "otsu"}}

    if algo_norm == "kmeans":
        labels2d, centers = seg_kmeans(
            img_bgr,
            k=int(params.get("k", 3)),
            attempts=int(params.get("attempts", 5)),
        )
        centers_gray = centers.mean(axis=1)
        fg_idx = int(np.argmin(centers_gray)) if params.get("fg_cluster") is None else int(params["fg_cluster"])
        mask01 = (labels2d == fg_idx).astype(np.uint8)
        overlay = overlay_mask(img_bgr, mask01, alpha=float(params.get("alpha", 0.5)))
        return {"mask01": mask01, "overlay": overlay, "meta": {"algo": "kmeans", "fg_cluster": fg_idx}}

    if algo_norm == "watershed":
        mask01 = seg_watershed(
            img_bgr,
            blur_ksize=int(params.get("blur_ksize", 5)),
            dist_thresh=float(params.get("dist_thresh", 0.4)),
            morph_ksize=int(params.get("morph_ksize", 3)),
        )
        overlay = overlay_mask(img_bgr, mask01, alpha=float(params.get("alpha", 0.5)))
        return {"mask01": mask01, "overlay": overlay, "meta": {"algo": "watershed"}}

    if algo_norm in ["deeplabv3_resnet50", "fcn_resnet50"]:
        dl = seg_torchvision(
            img_bgr,
            model_name=algo_norm,
            target_class=int(params.get("target_class", 15)),
            conf_thresh=float(params.get("conf_thresh", 0.5)),
            device=str(params.get("device", "cpu")),
        )
        overlay = overlay_mask(img_bgr, dl.mask01, alpha=float(params.get("alpha", 0.5)))
        return {
            "mask01": dl.mask01,
            "overlay": overlay,
            "meta": {"algo": algo_norm, "target_class": int(params.get("target_class", 15)), "num_classes": dl.num_classes},
        }

    if algo_norm in ["unet", "fpn"]:
        mask01 = seg_smp_binary(
            img_bgr,
            arch=algo_norm,
            encoder_name=str(params.get("encoder_name", "resnet34")),
            weights_bytes=params.get("weights_bytes", None),
            device=str(params.get("device", "cpu")),
            conf_thresh=float(params.get("conf_thresh", 0.5)),
        )
        overlay = overlay_mask(img_bgr, mask01, alpha=float(params.get("alpha", 0.5)))
        return {
            "mask01": mask01,
            "overlay": overlay,
            "meta": {"algo": algo_norm, "encoder": str(params.get("encoder_name", "resnet34"))},
        }

    raise ValueError(
        "Algoritma tidak dikenali. Pilih: otsu / kmeans / watershed / deeplabv3_resnet50 / fcn_resnet50 / unet / fpn"
    )
