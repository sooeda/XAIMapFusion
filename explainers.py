import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lime import lime_image
from scipy import ndimage as ndi


class ExplainersMixin:
    def make_fixed_segmentation_fn(self, segments):
        seg_for_lime = np.asarray(segments, dtype=np.int32).copy()
        seg_for_lime[seg_for_lime < 0] = 0

        def segmentation_fn(_img):
            return seg_for_lime

        return segmentation_fn

    def run_lime_pixel_map(self, model, image_np_for_model, fixed_segments, target_label=1, seed=None):
        seed = self.cfg.seed if seed is None else seed
        valid = fixed_segments >= 0
        if not np.any(valid):
            return np.zeros(image_np_for_model.shape[:2], dtype=np.float32)

        lime_segments = fixed_segments.copy()
        lime_segments[valid] += 1
        lime_segments[~valid] = 0
        n_features = int(lime_segments.max() + 1)

        explainer = lime_image.LimeImageExplainer(random_state=seed)
        seg_fn = self.make_fixed_segmentation_fn(lime_segments)

        kwargs = dict(
            image=image_np_for_model,
            classifier_fn=lambda arr: self.predict_proba_numpy(model=model, images_np=arr, batch_size=32),
            labels=(int(target_label),),
            top_labels=None,
            num_features=n_features,
            num_samples=self.cfg.lime_num_samples,
            batch_size=32,
            segmentation_fn=seg_fn,
            distance_metric="cosine",
        )

        sig = inspect.signature(explainer.explain_instance)
        if "random_seed" in sig.parameters:
            kwargs["random_seed"] = seed
        if "progress_bar" in sig.parameters:
            kwargs["progress_bar"] = False

        exp = explainer.explain_instance(**kwargs)

        raw_weights = np.zeros(n_features, dtype=np.float32)
        local_pairs = exp.local_exp[int(target_label)] if int(target_label) in exp.local_exp else []
        for seg_id, w in local_pairs:
            seg_id = int(seg_id)
            if 0 <= seg_id < n_features:
                raw_weights[seg_id] = float(w)

        region_weights = np.zeros(self.n_regions(fixed_segments), dtype=np.float32)
        for rid in range(self.n_regions(fixed_segments)):
            lime_id = rid + 1
            if lime_id < len(raw_weights):
                region_weights[rid] = raw_weights[lime_id]

        pixel_map = self.region_scores_to_map(fixed_segments, region_weights)
        return self.positive_importance_map(pixel_map)

    def _kernel_shap_forward(self, region_presence: torch.Tensor):
        if not hasattr(self, "_current_shap_context"):
            raise RuntimeError("SHAP context is not initialized.")
        ctx = self._current_shap_context

        if region_presence.ndim == 1:
            region_presence = region_presence.unsqueeze(0)

        presence_np = region_presence.detach().cpu().numpy()
        base_image = ctx["base_image"]
        original_image = ctx["original_image"]
        region_masks = ctx["region_masks"]
        target_label = ctx["target_label"]
        batch_images = []

        for sample_presence in presence_np:
            img = base_image.copy()
            active_ids = np.where(sample_presence > 0.5)[0]
            for rid in active_ids:
                mask = region_masks[rid]
                img[mask] = original_image[mask]
            batch_images.append(img)

        probs = self.predict_proba_numpy(self.model, batch_images, batch_size=32)
        out = torch.from_numpy(probs[:, int(target_label)].astype(np.float32)).to(region_presence.device)
        return out

    def run_shap_pixel_map(self, model, image_np_for_model, fixed_segments, target_label=1, seed=None):
        del model, seed
        n_regions = self.n_regions(fixed_segments)
        if n_regions == 0:
            return np.zeros(image_np_for_model.shape[:2], dtype=np.float32)

        region_masks = np.stack([(fixed_segments == rid) for rid in range(n_regions)], axis=0)
        fill = np.mean(image_np_for_model.reshape(-1, image_np_for_model.shape[-1]), axis=0).clip(0, 255).astype(np.uint8)
        base_image = np.zeros_like(image_np_for_model, dtype=np.uint8)
        base_image[...] = fill.reshape(1, 1, -1)

        self._current_shap_context = {
            "base_image": base_image,
            "original_image": image_np_for_model,
            "region_masks": region_masks,
            "target_label": int(target_label),
        }

        try:
            inputs = torch.ones((1, n_regions), dtype=torch.float32, device=self.device)
            baselines = torch.zeros((1, n_regions), dtype=torch.float32, device=self.device)
            attr = self.kernel_shap.attribute(
                inputs=inputs,
                baselines=baselines,
                n_samples=int(self.cfg.shap_num_samples),
                target=None,
            )
            region_weights = attr[0].detach().cpu().numpy().astype(np.float32)
        finally:
            if hasattr(self, "_current_shap_context"):
                del self._current_shap_context

        pixel_map = self.region_scores_to_map(fixed_segments, region_weights)
        return self.positive_importance_map(pixel_map)

    def find_last_conv_module(self, model):
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise RuntimeError("Не удалось найти Conv2d слой для CAM-метода.")
        return last_conv

    def add_small_noise_uint8(self, image_np, seed, sigma):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=image_np.shape).astype(np.float32)
        out = image_np.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    def run_gradcam_pixel_map(self, model, image_np_for_model, target_label=1):
        activations = []
        gradients = []
        target_module = self.find_last_conv_module(model)

        def fwd_hook(_m, _inp, out):
            activations.append(out.detach())

        def bwd_hook(_m, _gin, gout):
            gradients.append(gout[0].detach())

        h1 = target_module.register_forward_hook(fwd_hook)
        h2 = target_module.register_full_backward_hook(bwd_hook)

        try:
            model.zero_grad(set_to_none=True)
            x = self.preprocess_images_np(image_np_for_model[None, ...]).to(self.device)
            logits = model(x)
            score = logits[:, int(target_label)].sum()
            score.backward()

            if len(activations) == 0 or len(gradients) == 0:
                raise RuntimeError("Grad-CAM hooks did not capture tensors.")

            act = activations[0]
            grad = gradients[0]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        finally:
            h1.remove()
            h2.remove()

        return self.positive_importance_map(cam)

    def run_layercam_pixel_map(self, model, image_np_for_model, target_label=1):
        activations = []
        gradients = []
        target_module = self.find_last_conv_module(model)

        def fwd_hook(_m, _inp, out):
            activations.append(out.detach())

        def bwd_hook(_m, _gin, gout):
            gradients.append(gout[0].detach())

        h1 = target_module.register_forward_hook(fwd_hook)
        h2 = target_module.register_full_backward_hook(bwd_hook)

        try:
            model.zero_grad(set_to_none=True)
            x = self.preprocess_images_np(image_np_for_model[None, ...]).to(self.device)
            logits = model(x)
            score = logits[:, int(target_label)].sum()
            score.backward()

            if len(activations) == 0 or len(gradients) == 0:
                raise RuntimeError("Layer-CAM hooks did not capture tensors.")

            act = activations[0]
            grad = gradients[0]
            cam = torch.relu((torch.relu(grad) * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        finally:
            h1.remove()
            h2.remove()

        return self.positive_importance_map(cam)

    def run_smoothgrad_pixel_map(self, model, image_np_for_model, target_label=1, run_idx=0):
        del model
        base_x = self.preprocess_images_np(image_np_for_model[None, ...]).to(self.device)
        stdev = float(self.cfg.smoothgrad_noise_stds[min(run_idx, len(self.cfg.smoothgrad_noise_stds) - 1)])
        rng = np.random.default_rng(self.cfg.seed + 5000 + run_idx)

        maps = []
        for _ in range(self.cfg.smoothgrad_samples):
            noisy = base_x.clone()
            noise_np = rng.normal(0.0, stdev, size=noisy.shape).astype(np.float32)
            noise = torch.from_numpy(noise_np).to(self.device)
            noisy = noisy + noise
            noisy.requires_grad_(True)
            attr = self.saliency.attribute(noisy, target=target_label)
            sal = attr.abs().mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
            maps.append(sal)

        mean_map = np.mean(np.stack(maps, axis=0), axis=0)
        return self.positive_importance_map(mean_map)

    def build_integrated_gradients_baseline(self, image_np_for_model, base_x, baseline_mode: str):
        baseline_mode = str(baseline_mode).lower()

        if baseline_mode == "zero":
            return torch.zeros_like(base_x)

        if baseline_mode == "mean":
            return base_x.mean(dim=(2, 3), keepdim=True).expand_as(base_x).clone()

        if baseline_mode == "blur":
            img = image_np_for_model.astype(np.float32)
            blur_sigma = float(self.cfg.integrated_gradients_blur_sigma)
            blurred = ndi.gaussian_filter(img, sigma=(blur_sigma, blur_sigma, 0))
            blurred = np.clip(blurred, 0, 255).astype(np.uint8)
            return self.preprocess_images_np(blurred[None, ...]).to(self.device)

        return torch.zeros_like(base_x)

    def run_integrated_gradients_pixel_map(self, model, image_np_for_model, target_label=1, run_idx=0):
        del model
        x = self.preprocess_images_np(image_np_for_model[None, ...]).to(self.device)
        x.requires_grad_(True)

        baseline_modes = self.cfg.integrated_gradients_baseline_modes
        baseline_mode = baseline_modes[min(run_idx, len(baseline_modes) - 1)]
        baseline = self.build_integrated_gradients_baseline(image_np_for_model, x, baseline_mode)

        attr = self.integrated_gradients.attribute(
            inputs=x,
            baselines=baseline,
            target=int(target_label),
            n_steps=int(self.cfg.integrated_gradients_n_steps),
        )
        sal = attr.abs().mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
        return self.positive_importance_map(sal)
