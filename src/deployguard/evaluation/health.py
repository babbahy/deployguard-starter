from __future__ import annotations

import math

import torch
from transformers import TrainerCallback


def checked_probabilities(logits) -> torch.Tensor:
    logits = torch.as_tensor(logits)
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Non-finite model logits")
    probabilities = logits.float().softmax(dim=-1)
    if not torch.isfinite(probabilities).all().item():
        raise FloatingPointError("Non-finite model probabilities")
    if (probabilities < 0).any().item() or (probabilities > 1).any().item():
        raise FloatingPointError("Model probabilities outside [0, 1]")
    return probabilities


def check_finite_loss(loss: torch.Tensor) -> None:
    if loss is None or not torch.isfinite(loss.detach()).all().item():
        raise FloatingPointError("Non-finite training/evaluation loss")


class NumericalHealthCallback(TrainerCallback):
    """Fail closed on invalid gradients/parameters and retain magnitude maxima."""

    def __init__(self, max_gradient_abs: float = 1e6, max_parameter_abs: float = 1e6):
        self.max_gradient_abs = max_gradient_abs
        self.max_parameter_abs = max_parameter_abs
        self.diagnostics = {
            "finite_loss": True, "finite_gradients": True, "finite_parameters": True,
            "max_gradient_abs": 0.0, "max_parameter_abs": 0.0,
            "gradient_checks": 0, "parameter_checks": 0,
            "gradient_magnitude_limit": max_gradient_abs,
            "parameter_magnitude_limit": max_parameter_abs,
        }

    @staticmethod
    def _max_abs(tensors: list[torch.Tensor], name: str) -> float:
        if not tensors:
            return 0.0
        if any(not torch.isfinite(t).all().item() for t in tensors):
            raise FloatingPointError(f"Non-finite {name}")
        maxima = torch.stack([t.detach().abs().max().float() for t in tensors])
        return float(maxima.max().item())

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        maximum = self._max_abs(gradients, "gradient")
        self.diagnostics["gradient_checks"] += 1
        self.diagnostics["max_gradient_abs"] = max(
            self.diagnostics["max_gradient_abs"], maximum
        )
        if maximum > self.max_gradient_abs:
            self.diagnostics["finite_gradients"] = False
            raise FloatingPointError(f"Gradient magnitude {maximum:g} exceeds configured limit")
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        parameters = [p.detach() for p in model.parameters()]
        maximum = self._max_abs(parameters, "parameter")
        self.diagnostics["parameter_checks"] += 1
        self.diagnostics["max_parameter_abs"] = max(
            self.diagnostics["max_parameter_abs"], maximum
        )
        if maximum > self.max_parameter_abs:
            self.diagnostics["finite_parameters"] = False
            raise FloatingPointError(f"Parameter magnitude {maximum:g} exceeds configured limit")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key in ("loss", "eval_loss"):
                value = logs.get(key)
                if value is not None and (not math.isfinite(float(value))):
                    self.diagnostics["finite_loss"] = False
                    raise FloatingPointError(f"Non-finite {key}: {value}")
        return control
