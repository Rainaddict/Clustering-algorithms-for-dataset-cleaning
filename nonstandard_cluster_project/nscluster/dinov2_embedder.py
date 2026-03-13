from __future__ import annotations
import torch
from torchvision import transforms
from PIL import Image


class DinoV2Embedder:
    """
    用 torch.hub 加载 DINOv2：
      dinov2_vitb14 (768)
      dinov2_vitl14 (1024)
    输出：L2-normalized CLS embedding
    """
    def __init__(self, model_name: str, input_size: int, device: str = "cuda"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.input_size = input_size

        # load model
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        # infer dim
        dummy = torch.zeros(1, 3, input_size, input_size, device=self.device)
        with torch.no_grad():
            feat = self._forward(dummy)
        self.dim = int(feat.shape[-1])

        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 的 forward_features 返回 dict，取 x_norm_clstoken
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            # 兜底：有些版本键名可能不同
            for k in ["x_clstoken", "cls_token", "x_norm_cls_token"]:
                if k in out:
                    return out[k]
            raise KeyError(f"Unsupported dinov2 forward_features keys: {list(out.keys())}")
        # 若直接返回 tensor
        return out

    def encode(self, pil_images: list[Image.Image]) -> torch.Tensor:
        xs = torch.stack([self.tf(im) for im in pil_images], dim=0).to(self.device, non_blocking=True)
        feat = self._forward(xs)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat