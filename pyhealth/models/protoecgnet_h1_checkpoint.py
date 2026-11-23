"""Bind H1 projection sources to the bank and feature state that produced them.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Uses the existing standard projection algorithm. This is native H1 checkpoint
support, not a converter for historical reference checkpoints.
"""

import copy
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .protoecgnet import ProtoECGNet
from .protoecgnet_h1_data import _integer

if TYPE_CHECKING:
    from .protoecgnet_h1 import ProtoECGNetH1

PREFIX = "branches.morphology."


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def _tensor_bytes(value: Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    header = _json_bytes({"dtype": str(value.dtype), "shape": list(value.shape)})
    return header + b"\0" + value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_digest(value: Tensor) -> str:
    return hashlib.sha256(_tensor_bytes(value)).hexdigest()


def _bank_tensors(state: Mapping[str, Any]) -> dict[str, Tensor]:
    names = [
        name
        for name in state
        if name.startswith((PREFIX + "backbone.", PREFIX + "add_on."))
    ]
    names += [PREFIX + "prototype_vectors", PREFIX + "prototype_class_identity"]
    if len(names) < 4 or any(
        name not in state or not isinstance(state[name], Tensor) for name in names
    ):
        raise ValueError("A complete morphology bank and feature state is required.")
    return {name: state[name] for name in sorted(names)}


def _live_tensors(model: "ProtoECGNetH1") -> dict[str, Tensor]:
    # Do not call state_dict here: get_extra_state itself uses this fingerprint.
    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def bank_state_digest(
    state: Mapping[str, Any],
    label_names: list[str],
    architecture: dict[str, Any],
) -> str:
    """Hash prototypes/assignments, feature weights/BN buffers, labels and config.

    Classifier weights are excluded so classifier-only refit keeps the binding.
    Values, dtypes and shapes are hashed on CPU independently of tensor device.
    """
    digest = hashlib.sha256(
        _json_bytes(
            {"format_version": 1, "labels": label_names, "architecture": architecture}
        )
    )
    for name, tensor in _bank_tensors(state).items():
        digest.update(name.encode() + b"\0" + _tensor_bytes(tensor))
    return digest.hexdigest()


def _check_math(model: "ProtoECGNetH1") -> None:
    branch = model.branches["morphology"]
    actual = (
        branch.similarity,
        branch.window,
        branch.proto_time_len,
        branch.topk,
        branch.epsilon,
        branch.scale,
    )
    expected = ("cosine", 3, 3, 5, 1e-4, math.isqrt(branch.proto_dim * 3) / 3)
    if actual != expected:
        raise ValueError(
            "H1 morphology settings differ from the fixed reference computation."
        )


def validate_projection_provenance(
    provenance: Mapping[str, Any],
    state: Mapping[str, Any],
    label_names: list[str],
    architecture: dict[str, Any],
) -> None:
    """Reject stale/mismatched bank bindings and inconsistent source records.

    This checks checkpoint integrity. Source input hashes bind the exact
    preprocessed waveforms seen during native projection; input reconstruction
    against an external dataset is a separate verification step.
    """
    if not isinstance(provenance, dict) or provenance.get("format_version") != 1:
        raise ValueError("Unsupported projection provenance format.")
    if (
        provenance.get("label_names") != label_names
        or provenance.get("architecture") != architecture
    ):
        raise ValueError("Projection provenance label order or architecture mismatch.")
    if provenance.get("bank_state_sha256") != bank_state_digest(
        state, label_names, architecture
    ):
        raise ValueError(
            "Projection binding is stale or belongs to another bank/feature state."
        )
    records = provenance.get("source_records")
    bank = state[PREFIX + "prototype_vectors"]
    own = state[PREFIX + "prototype_class_identity"].detach().cpu()
    if not isinstance(records, list) or len(records) != len(bank):
        raise ValueError("Projection source count must match the prototype bank.")
    payload = {
        key: value for key, value in provenance.items() if key != "source_bundle_sha256"
    }
    if (
        provenance.get("source_bundle_sha256")
        != hashlib.sha256(_json_bytes(payload)).hexdigest()
    ):
        raise ValueError("Projection source bundle checksum mismatch.")
    covered = 0
    for prototype_id, record in enumerate(records):
        if record is None:
            continue
        covered += 1
        if not isinstance(record, dict) or record.get("prototype_id") != prototype_id:
            raise ValueError("Projection records are not aligned to prototype indices.")
        assignment = own[prototype_id]
        if (
            not bool(((assignment == 0) | (assignment == 1)).all())
            or assignment.sum().item() != 1
        ):
            raise ValueError("H1 projection requires one owned label per prototype.")
        label = label_names[int(assignment.argmax())]
        if record.get("assigned_label") != label or label not in record.get(
            "source_labels", []
        ):
            raise ValueError("Projection source does not carry the assigned label.")
        source_id = _integer(record.get("source_ecg_id"), "source_ecg_id")
        if source_id < 0 or _integer(
            record.get("source_fold"), "source_fold"
        ) not in range(1, 9):
            raise ValueError("Projection sources must come from training folds 1-8.")
        if (
            not isinstance(record.get("source_patient_id"), str)
            or not record["source_patient_id"]
        ):
            raise ValueError("Projection source patient ID is missing.")
        start = _integer(record.get("latent_start"), "latent_start")
        windows = _integer(record.get("latent_num_windows"), "latent_num_windows")
        if (
            start >= windows
            or record.get("latent_end_exclusive") != start + architecture["window"]
        ):
            raise ValueError("Projection latent-window coordinates are inconsistent.")
        if (
            record.get("window") != start
            or _integer(record.get("sample"), "sample") >= provenance["input_records"]
        ):
            raise ValueError("Projection loader-relative coordinates are inconsistent.")
        if record.get("prototype_vector_sha256") != _tensor_digest(bank[prototype_id]):
            raise ValueError(
                "Projection record is linked to a different prototype vector."
            )
        if (
            not isinstance(record.get("source_signal_sha256"), str)
            or len(record["source_signal_sha256"]) != 64
        ):
            raise ValueError("Projection source input fingerprint is missing.")
        score = record.get("pre_projection_similarity")
        if (
            not isinstance(score, (float, int))
            or not math.isfinite(score)
            or record.get("similarity") != score
        ):
            raise ValueError("Projection similarity is invalid.")
    if provenance.get("covered_prototypes") != covered:
        raise ValueError("Projection coverage does not match the source records.")


def checked_projection_provenance(model: "ProtoECGNetH1") -> dict[str, Any] | None:
    """Return a validated copy, or None when no sources are bound."""
    provenance = getattr(model, "_projection_provenance", None)
    if provenance is None:
        return None
    _check_math(model)
    validate_projection_provenance(
        provenance, _live_tensors(model), model.label_names, model._architecture()
    )
    return copy.deepcopy(provenance)


class _ProjectionLoader:
    """Capture IDs/input hashes from the exact pass used by standard projection."""

    def __init__(self, loader: Any, model: "ProtoECGNetH1") -> None:
        self.loader, self.model = loader, model
        self.sampler = getattr(loader, "sampler", None)
        self.shuffle_samples = getattr(loader, "shuffle_samples", False)
        self.rows: list[dict[str, Any]] = []
        self.pending: tuple[int, int] | None = None

    def capture_latent_shape(
        self,
        module: torch.nn.Module,
        inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        if self.pending is None or len(output) != self.pending[1]:
            raise ValueError(
                "Unexpected feature pass during projection source tracking."
            )
        start, count = self.pending
        windows = int(output.shape[-1]) - self.model.branches["morphology"].window + 1
        for row in self.rows[start : start + count]:
            row["latent_num_windows"] = windows
        self.pending = None

    def __iter__(self) -> Iterator[dict[str, Any]]:
        previous_id = -1
        for batch in self.loader:
            required = (
                "ecg_id",
                "patient_id",
                "strat_fold",
                self.model.signal_key,
                self.model.label_key,
            )
            if any(key not in batch for key in required):
                raise ValueError(
                    "Projection needs ECG/patient IDs, folds, signals and targets."
                )
            x = batch[self.model.signal_key]
            y = batch[self.model.label_key].detach().cpu()
            if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
                raise ValueError(
                    "Projection inputs must be finite preprocessed float32 ECGs."
                )
            count = len(x)
            if (
                count == 0
                or y.shape != (count, self.model.num_labels)
                or not bool(((y == 0) | (y == 1)).all())
            ):
                raise ValueError(
                    "Projection targets must be nonempty binary multi-hot labels."
                )
            if any(
                len(batch[key]) != count
                for key in ("ecg_id", "patient_id", "strat_fold")
            ):
                raise ValueError(
                    "Projection metadata length does not match the ECG batch."
                )
            start = len(self.rows)
            for index in range(count):
                ecg_id = _integer(batch["ecg_id"][index], "ecg_id")
                fold = _integer(batch["strat_fold"][index], "fold")
                patient = batch["patient_id"][index]
                if patient is None or not str(patient).strip():
                    raise ValueError("Projection source patient ID is missing.")
                if ecg_id <= previous_id:
                    raise ValueError(
                        "Projection ECG IDs must be unique and strictly increasing."
                    )
                if fold not in range(1, 9):
                    raise ValueError(
                        "Projection sources must come from training folds 1-8."
                    )
                previous_id = ecg_id
                # Match the image/lead layout actually supplied to the backbone.
                signal = x[index].detach().cpu()
                if signal.ndim == 3 and signal.shape[0] == 1:
                    signal = signal.squeeze(0)
                self.rows.append(
                    {
                        "ecg_id": ecg_id,
                        "patient_id": str(patient),
                        "fold": fold,
                        "labels": [
                            self.model.label_names[column]
                            for column in y[index].nonzero().flatten().tolist()
                        ],
                        "signal_sha256": _tensor_digest(signal),
                    }
                )
            self.pending = (start, count)
            yield batch
            if self.pending is not None:
                raise ValueError(
                    "Projection did not consume the tracked feature batch."
                )
        if not self.rows:
            raise ValueError("Projection loader is empty.")


@torch.no_grad()
def project_with_sources(
    model: "ProtoECGNetH1",
    loader: Any,
    *,
    source_context: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any] | None]]:
    """Run standard projection and bind training-only sources to the new bank.

    Loader order must be unique ECG ID ascending; exact-score ties keep the
    earliest ECG/window through the unchanged base algorithm. Metadata and
    source hashes are captured from this same pass, not a second iteration.
    Context is JSON-compatible declared preprocessing/protocol metadata, not
    a certificate that an external dataset has been reconstructed.
    """
    _check_math(model)
    context = json.loads(_json_bytes({} if source_context is None else source_context))
    tracked = _ProjectionLoader(loader, model)
    branch = model.branches["morphology"]
    old_bank = branch.prototype_vectors.detach().clone()
    old_info = copy.deepcopy(model.projection_info)
    old_provenance = copy.deepcopy(model._projection_provenance)
    was_training = model.training
    hook = branch.backbone.register_forward_hook(tracked.capture_latent_shape)
    try:
        info = ProtoECGNet.project_prototypes(model, tracked)
        records = []
        for prototype_id, entry in enumerate(info["morphology"]):
            if entry is None:
                records.append(None)
                continue
            row = tracked.rows[entry["sample"]]
            label = model.label_names[
                int(branch.prototype_class_identity[prototype_id].argmax())
            ]
            records.append(
                {
                    **entry,
                    "prototype_id": prototype_id,
                    "assigned_label": label,
                    "source_ecg_id": row["ecg_id"],
                    "source_patient_id": row["patient_id"],
                    "source_fold": row["fold"],
                    "source_labels": row["labels"],
                    "source_signal_sha256": row["signal_sha256"],
                    "latent_start": entry["window"],
                    "latent_end_exclusive": entry["window"] + branch.window,
                    "latent_num_windows": row["latent_num_windows"],
                    "pre_projection_similarity": entry["similarity"],
                    "prototype_vector_sha256": _tensor_digest(
                        branch.prototype_vectors[prototype_id]
                    ),
                }
            )
        provenance = {
            "format_version": 1,
            "label_names": list(model.label_names),
            "architecture": model._architecture(),
            "source_records": records,
            "covered_prototypes": sum(record is not None for record in records),
            "input_records": len(tracked.rows),
            "input_manifest_sha256": hashlib.sha256(
                _json_bytes(tracked.rows)
            ).hexdigest(),
            "source_context": context,
            "tie_break": "Exact similarity ties: ascending ECG ID, then latent window",
            "bank_state_sha256": bank_state_digest(
                _live_tensors(model), model.label_names, model._architecture()
            ),
        }
        provenance["source_bundle_sha256"] = hashlib.sha256(
            _json_bytes(provenance)
        ).hexdigest()
        validate_projection_provenance(
            provenance, _live_tensors(model), model.label_names, model._architecture()
        )
        model._projection_provenance = provenance
        model.projection_info = {"morphology": copy.deepcopy(records)}
        return copy.deepcopy(model.projection_info)
    except Exception:
        branch.prototype_vectors.copy_(old_bank)
        model.projection_info = old_info
        model._projection_provenance = old_provenance
        raise
    finally:
        hook.remove()
        model.train(was_training)
