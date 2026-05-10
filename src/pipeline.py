import torch
import numpy as np
import cv2

from .models import MultiTaskClassifier, FractureSegmenter
from .transforms import get_inference_cls_transform, get_inference_seg_transform

BONE_CLASSES = ['Hand', 'Leg', 'Hip', 'Shoulder', 'Mixed']


class CascadePipeline:
    """
    Two-stage inference pipeline:
      Stage 1 — MultiTaskClassifier: bone type + fracture probability
      Stage 2 — FractureSegmenter:   pixel-wise mask (only if prob > threshold)
    """

    def __init__(self, classifier_path: str, segmenter_path: str, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classifier = MultiTaskClassifier(num_bone_classes=5).to(self.device)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=self.device, weights_only=True)
        )
        self.classifier.eval()

        self.segmenter = FractureSegmenter(in_channels=3, out_channels=1).to(self.device)
        self.segmenter.load_state_dict(
            torch.load(segmenter_path, map_location=self.device, weights_only=True)
        )
        self.segmenter.eval()

        self._t_cls = get_inference_cls_transform()
        self._t_seg = get_inference_seg_transform()

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Args:
            img_rgb:   H×W×3 uint8 RGB numpy array
            threshold: fracture probability cut-off

        Returns:
            bone_type    (str)   — detected bone label(s)
            bone_probs   (dict)  — {label: float probability}
            fracture_prob (float)
            is_fractured (bool)
            mask         (H×W uint8 ndarray or None)
        """
        h, w = img_rgb.shape[:2]

        t_cls = self._t_cls(image=img_rgb)['image'].unsqueeze(0).to(self.device)
        t_seg = self._t_seg(image=img_rgb)['image'].unsqueeze(0).to(self.device)

        # ── Stage 1 ──────────────────────────────────────────────────────
        logits_bone, logits_frac = self.classifier(t_cls)
        fracture_prob = float(torch.sigmoid(logits_frac).item())

        bone_probs_arr = torch.sigmoid(logits_bone).squeeze().cpu().numpy()
        bone_probs = {cls: float(p) for cls, p in zip(BONE_CLASSES, bone_probs_arr)}

        # Exclude 'Mixed' from individual detection — treat it as derived
        individual = [cls for cls, p in bone_probs.items() if p > 0.5 and cls != 'Mixed']
        if len(individual) > 1:
            bone_type = f"Mixed ({', '.join(individual)})"
        elif len(individual) == 1:
            bone_type = individual[0]
        else:
            # Nothing above 0.5 — take argmax among non-Mixed classes, or Mixed if it wins
            bone_type = BONE_CLASSES[int(np.argmax(bone_probs_arr))]

        # ── Stage 2 ──────────────────────────────────────────────────────
        mask = None
        if fracture_prob > threshold:
            logits_mask = self.segmenter(t_seg)
            mask_prob   = torch.sigmoid(logits_mask).squeeze().cpu().numpy()
            mask_bin    = (mask_prob > 0.5).astype(np.uint8)
            mask = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        return {
            'bone_type':     bone_type,
            'bone_probs':    bone_probs,
            'fracture_prob': fracture_prob,
            'is_fractured':  fracture_prob > threshold,
            'mask':          mask,
        }
