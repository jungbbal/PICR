"""Batched open-loop prediction shared by the evaluation scripts."""
from __future__ import annotations

import torch


@torch.inference_mode()
def predict_batch(model, batch: dict, device: torch.device, horizon: int) -> torch.Tensor:
    """(B, horizon, state_dim) predictions for one dataset batch."""
    history_states = batch["history_states"].to(device)
    history_torques = batch["history_torques"].to(device)
    future_torques = batch["future_torques"].to(device)

    return model(
        history_states,
        future_torques,
        batch["object_context"].to(device),
        history_torques=history_torques if getattr(model, "use_context_encoder", False) else None,
        return_aux=False,
    )
