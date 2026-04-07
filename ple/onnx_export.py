"""
ONNX Export — CPU deployment for Oracle server

Exports trained PLE v4/v6 models to ONNX format for:
  - CPU inference without PyTorch dependency
  - 2-5x faster inference via ONNX Runtime
  - 24GB RAM server: lightweight deployment

Usage:
  # Export
  python -m ple.onnx_export --model models/production_v5/bchusdt.pt --output models/onnx/bchusdt.onnx

  # Inference
  session = InferenceSession("models/onnx/bchusdt.onnx")
  result = session.run(None, {"features": features, "account": account})
"""

import os
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from pathlib import Path


def export_v4_to_onnx(
    model,
    feature_dim: int,
    n_account: int = 4,
    output_path: str = "model.onnx",
    opset_version: int = 17,
):
    """Export PLE v4 model to ONNX."""
    model.eval()
    device = next(model.parameters()).device

    # Dummy inputs
    dummy_features = torch.randn(1, feature_dim, device=device)
    dummy_account = torch.randn(1, n_account, device=device)

    # Export
    torch.onnx.export(
        model,
        (dummy_features, dummy_account),
        output_path,
        input_names=["features", "account"],
        output_names=["label_probs", "mae_pred", "mfe_pred", "confidence"],
        dynamic_axes={
            "features": {0: "batch"},
            "account": {0: "batch"},
            "label_probs": {0: "batch"},
            "mae_pred": {0: "batch"},
            "mfe_pred": {0: "batch"},
            "confidence": {0: "batch"},
        },
        opset_version=opset_version,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Verify outputs match
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy_features.cpu(), dummy_account.cpu())

    ort_out = session.run(None, {
        "features": dummy_features.cpu().numpy(),
        "account": dummy_account.cpu().numpy(),
    })

    torch_probs = torch_out["label_probs"].cpu().numpy()
    onnx_probs = ort_out[0]
    max_diff = np.abs(torch_probs - onnx_probs).max()
    print(f"  ONNX exported: {output_path}")
    print(f"  Max output difference: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Output mismatch: {max_diff}"
    return output_path


def export_v6_to_onnx(
    model,
    feature_dim: int,
    n_account: int = 4,
    n_temporal: int = 40,
    output_path: str = "model_v6.onnx",
    opset_version: int = 17,
):
    """Export PLE v6 model to ONNX."""
    model.eval()
    model = model.cpu()

    # Dummy inputs
    dummy_features = torch.randn(1, feature_dim)
    dummy_coin_id = torch.tensor([0], dtype=torch.long)
    dummy_regime_id = torch.tensor([2], dtype=torch.long)  # range
    dummy_account = torch.randn(1, n_account)
    dummy_temporal = torch.randn(1, n_temporal)

    # Export
    torch.onnx.export(
        model,
        (dummy_features, dummy_coin_id, dummy_regime_id, dummy_account, dummy_temporal),
        output_path,
        input_names=["features", "coin_id", "regime_id", "account", "temporal"],
        output_names=["label_probs", "mae_pred", "mfe_pred", "confidence", "leverage_rec"],
        dynamic_axes={
            "features": {0: "batch"},
            "coin_id": {0: "batch"},
            "regime_id": {0: "batch"},
            "account": {0: "batch"},
            "temporal": {0: "batch"},
            "label_probs": {0: "batch"},
            "mae_pred": {0: "batch"},
            "mfe_pred": {0: "batch"},
            "confidence": {0: "batch"},
            "leverage_rec": {0: "batch"},
        },
        opset_version=opset_version,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX v6 exported: {output_path}")
    return output_path


class ONNXInference:
    """Lightweight inference wrapper for production deployment.

    Designed for Oracle server (24GB RAM, CPU only):
      - Loads ONNX model once
      - Minimal memory footprint
      - Thread-safe inference
    """

    def __init__(self, model_path: str, config_path: str | None = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=self._get_options(),
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

    def _get_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # Oracle server: tune to available cores
        opts.inter_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def predict(self, features: np.ndarray, account: np.ndarray | None = None) -> dict:
        """Run inference. Returns dict with label_probs, mae_pred, mfe_pred, confidence."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if account is None:
            account = np.zeros((features.shape[0], 4), dtype=np.float32)
            account[:, 0] = 1.0  # normalized equity

        inputs = {
            "features": features.astype(np.float32),
            "account": account.astype(np.float32),
        }

        outputs = self.session.run(None, inputs)

        result = {}
        for name, arr in zip(self.output_names, outputs):
            result[name] = arr

        return result

    def predict_v6(
        self,
        features: np.ndarray,
        coin_id: int = 0,
        regime_id: int = 2,  # range default
        account: np.ndarray | None = None,
        temporal: np.ndarray | None = None,
    ) -> dict:
        """Run v6 inference with regime and coin context."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        B = features.shape[0]
        if account is None:
            account = np.zeros((B, 4), dtype=np.float32)
            account[:, 0] = 1.0
        if temporal is None:
            temporal = np.zeros((B, 40), dtype=np.float32)

        inputs = {
            "features": features.astype(np.float32),
            "coin_id": np.full(B, coin_id, dtype=np.int64),
            "regime_id": np.full(B, regime_id, dtype=np.int64),
            "account": account.astype(np.float32),
            "temporal": temporal.astype(np.float32),
        }

        outputs = self.session.run(None, inputs)
        result = {}
        for name, arr in zip(self.output_names, outputs):
            result[name] = arr
        return result

    def benchmark(self, feature_dim: int, n_runs: int = 1000) -> dict:
        """Benchmark inference latency."""
        import time
        features = np.random.randn(1, feature_dim).astype(np.float32)
        account = np.zeros((1, 4), dtype=np.float32)
        account[0, 0] = 1.0

        # Warmup
        for _ in range(10):
            self.predict(features, account)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.predict(features, account)
        elapsed = time.perf_counter() - t0

        return {
            "total_ms": elapsed * 1000,
            "per_inference_ms": elapsed / n_runs * 1000,
            "inferences_per_sec": n_runs / elapsed,
        }
