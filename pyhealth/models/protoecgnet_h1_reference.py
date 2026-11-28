"""Explicit import of the pinned, refitted H1 research checkpoint format.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., MLHC 2025.
ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG
Classification with Contrastive Learning.
https://proceedings.mlr.press/v298/sethi25a.html

No file is unpickled here. Callers must authenticate trusted input files before
loading them. Conversion does not project, train, or restore optimizer state.
"""

import copy
import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

from .protoecgnet_h1 import ProtoECGNetH1
from .protoecgnet_h1_checkpoint import (
    PREFIX,
    _check_math,
    _integer,
    _json_bytes,
    _live_tensors,
    _tensor_digest,
    bank_state_digest,
    validate_projection_provenance,
)

REFERENCE_COMMIT = "fefbe6166a87a961c9c65d220d6344e8d53f2c49"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def convert_reference_checkpoint(
    model: ProtoECGNetH1, checkpoint: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete strict-load state without changing the destination model.

    Supports the refitted, class-specific morphology research format only.
    Label order, original objective settings, assignment, statistics, and every
    active tensor are checked. Unknown source keys fail closed. The only unused
    author tensors are the legacy ``ones`` bank and dormant ResNet FC weights.
    Destination pretraining-head tensors are unused in joint/refit and retained.

    Returns:
        Native state_dict and an explicit mapping/import record. Original source
        records are copied to that record but NOT asserted as verified native
        provenance. Call ``bind_reconstructed_reference_sources`` to reconstruct
        all saved source patches before persisting bank-linked provenance.

    Args:
        model: Destination native H1 model with the same architecture and label order.
        checkpoint: Trusted, already-loaded epoch-35 research refit checkpoint payload.

    Raises:
        ValueError: Architecture, labels, statistics, roles or source tensors differ.
        KeyError: The supported research checkpoint schema is incomplete.

    Examples:
        >>> native, mapping = convert_reference_checkpoint(  # doctest: +SKIP
        ...     model, trusted_reference_checkpoint)
        >>> model.load_state_dict(native, strict=True)  # doctest: +SKIP
    """
    _check_math(model)
    spec = checkpoint["run_spec"]
    branch = model.branches["morphology"]
    count, width, window = branch.prototype_vectors.shape
    labels = list(model.label_names)
    _require(
        checkpoint.get("stage") == "h1_refitted", "Expected a refitted checkpoint."
    )
    _require(checkpoint.get("selection_role") == "best", "Unexpected selection role.")
    _require(checkpoint.get("completed_refit_epochs") == 35, "Unexpected refit epoch.")
    arm = checkpoint.get("arm")
    coefficients = {"paired_lambda0": 0.0, "paired_consistency": 117608.0}
    _require(arm in coefficients, "Unknown research arm.")
    protocol = checkpoint["h1_training_protocol"]
    _require(protocol["lambda_by_arm"] == coefficients, "Locked coefficients differ.")
    _require(spec["reference_commit"] == REFERENCE_COMMIT, "Reference commit differs.")
    _require(spec["label_names"] == labels, "Ordered labels differ.")
    _require(spec["pooling"] == "top5_mean", "Reference pooling differs.")
    expected_config = {
        "backbone": "resnet18",
        "num_classes": len(labels),
        "single_class_prototype_per_class": count // len(labels),
        "joint_prototypes_per_border": 0,
        "proto_dim": width,
        "proto_time_len": window,
        "latent_space_type": "arc",
        "add_on_layers_type": "linear",
        "last_layer_connection_weight": 1.0,
        "custom_groups": True,
        "label_set": "3",
        "pretrained_weights": None,
    }
    _require(spec["model_config"] == expected_config, "Reference architecture differs.")
    losses = {
        "lam_clst": 0.004,
        "lam_sep": 0.0004,
        "lam_spars": 0.0,
        "lam_div": 250.0,
        "lam_cnrst": 300.0,
        "use_contrastive": True,
    }
    _require(spec["loss_settings"] == losses, "Original objective settings differ.")
    _require(
        protocol["run_spec"]["model_config"] == expected_config
        and protocol["run_spec"]["loss_settings"] == losses,
        "Checkpoint protocol and model settings differ.",
    )
    sources = copy.deepcopy(checkpoint["projection_sources"])
    _require(len(sources) == count, "Reference source count differs.")
    train_ids = spec["train_ecg_ids"]
    _require(len(set(train_ids)) == len(train_ids), "Duplicate training ECG IDs.")
    train_set = set(train_ids)
    expected_assignment = torch.eye(len(labels)).repeat_interleave(
        count // len(labels), 0
    )
    source = checkpoint["model_state_dict"]
    _require(
        all(
            isinstance(t, Tensor) and bool(torch.isfinite(t).all())
            for t in source.values()
        ),
        "Non-tensor or nonfinite reference state.",
    )
    _require(
        torch.equal(source["prototype_class_identity"].cpu(), expected_assignment),
        "Reference assignment differs.",
    )
    _require(
        torch.equal(checkpoint["prototype_class_identity"].cpu(), expected_assignment),
        "Top-level reference assignment differs.",
    )
    for index, record in enumerate(sources):
        _require(
            record["prototype_id"] == index
            and record["assigned_label"] == labels[index // (count // len(labels))],
            "Reference source order or assignment differs.",
        )
        _require(
            record["source_ecg_id"] in train_set, "Source is outside training IDs."
        )
        start = _integer(record["latent_start"], "latent_start")
        _require(
            record["latent_end_exclusive"] == start + window
            and math.isfinite(record["pre_projection_similarity"]),
            "Invalid source coordinate or similarity.",
        )
    weights = checkpoint["class_weights"]
    cooc = source["label_cooccurrence"]
    _require(
        weights.shape == (len(labels),)
        and weights.dtype == torch.float32
        and bool(torch.isfinite(weights).all())
        and bool((weights > 0).all()),
        "Invalid training class weights.",
    )
    _require(
        cooc.shape == (len(labels), len(labels))
        and cooc.dtype == torch.float32
        and bool(((cooc >= 0) & (cooc <= 1)).all())
        and torch.allclose(cooc, cooc.t()),
        "Invalid training co-occurrence.",
    )
    state = copy.deepcopy(model.state_dict())
    mapping = {}
    consumed = set()

    def assign(destination: str, origin: str, squeeze: int | None = None) -> None:
        value = source[origin].detach().cpu()
        if squeeze is not None:
            _require(
                value.ndim == 4 and value.shape[squeeze] == 1,
                f"Expected singleton reference axis: {origin}",
            )
            value = value.squeeze(squeeze)
        target = state[destination]
        _require(
            value.shape == target.shape and value.dtype == target.dtype,
            f"Tensor shape/dtype differs: {origin} -> {destination}",
        )
        state[destination] = value.clone()
        mapping[destination] = origin
        consumed.add(origin)

    for name in branch.backbone.net.state_dict():
        assign(PREFIX + "backbone.net." + name, "feature_extractor.resnet." + name)
    for name in branch.add_on.state_dict():
        assign(
            PREFIX + "add_on." + name,
            "add_on_layers." + name,
            -1 if name.endswith("weight") else None,
        )
    assign(PREFIX + "prototype_vectors", "prototype_vectors", 2)
    assign(PREFIX + "prototype_class_identity", "prototype_class_identity")
    assign(PREFIX + "classifier.weight", "classifier.weight")
    assign("label_cooccurrence", "label_cooccurrence")
    state["class_weights"] = weights.detach().cpu().clone()
    # Rebuild label-derived routing instead of retaining a caller's old values.
    state["index_morphology"] = torch.arange(len(labels), dtype=torch.int64)
    state["fusion_mask"] = expected_assignment.t().contiguous()
    inactive = {
        "ones": "Legacy activation helper; cosine path does not use it.",
        "feature_extractor.resnet.fc.weight": "Dormant FC; conv_features returns spatial features.",
        "feature_extractor.resnet.fc.bias": "Dormant FC; conv_features returns spatial features.",
    }
    _require(
        set(source) - consumed == set(inactive),
        "Unknown or missing inactive source keys.",
    )
    _require(
        source["ones"].shape == (count, width, 1, window)
        and source["ones"].dtype == torch.float32
        and bool((source["ones"] == 1).all()),
        "Legacy ones bank differs.",
    )
    _require(
        source["feature_extractor.resnet.fc.weight"].shape == (1000, 512)
        and source["feature_extractor.resnet.fc.bias"].shape == (1000,),
        "Dormant reference FC shape differs.",
    )
    extra = state["_extra_state"]
    extra["stage"] = "classifier"
    extra["objective"] = {
        **{k: v for k, v in losses.items() if k != "use_contrastive"},
        "lambda_cons": coefficients[arm],
        "lam_l1": 1e-4,
    }
    extra["contrastive_enabled"] = True
    extra["projection_provenance"] = None
    retained = [
        name
        for name in state
        if name not in mapping and name != "class_weights" and name != "_extra_state"
    ]
    _require(
        all(
            name in ("_dummy_param", "fusion_mask", "index_morphology")
            or name.startswith(PREFIX + "pretrain_head.")
            for name in retained
        ),
        "Unmapped active destination tensors.",
    )
    return state, {
        "format_version": 1,
        "arm": arm,
        "reference_commit": REFERENCE_COMMIT,
        "source_to_destination": {origin: dest for dest, origin in mapping.items()},
        "inactive_source_keys": inactive,
        "retained_destination_keys": retained,
        "retained_destination_scope": "Unused pretraining head/empty device marker; label-derived index and fusion mask.",
        "class_weights_origin": "checkpoint.class_weights",
        "projection_sources": sources,
        "projection_sources_bound": False,
        "optimizer_restored": False,
    }


@torch.no_grad()
def bind_reconstructed_reference_sources(
    model: ProtoECGNetH1,
    sources: list[dict[str, Any]],
    source_batches: Iterable[dict[str, Any]],
    *,
    source_context: dict[str, Any],
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> dict[str, Any]:
    """Verify all saved patches and bind their identities without reprojection.

    ``source_batches`` must contain exactly the unique saved-source ECGs, with
    preprocessed float32 signal, binary targets, ecg_id, patient_id, strat_fold.
    Source batches may be in any order. All records must be from folds 1-8 and
    carry their assigned label. Input fingerprints cover these source ECGs,
    not an assertion that the full original projection search was rerun.

    Tolerances must be declared before running an external audit. A failure
    leaves model weights, BN buffers, mode and existing provenance unchanged.
    Saved pre-projection similarity is retained as a historical value, not
    recomputed with the projected bank. No optimizer state is imported.

    Args:
        model: Eval-mode H1 refit model with the already-imported fixed bank.
        sources: Saved prototype-ordered historical source records.
        source_batches: Batches for exactly the unique saved-source training ECGs.
        source_context: JSON-compatible input/protocol identities to persist.
        atol: Fixed absolute tolerance for source-patch reconstruction.
        rtol: Fixed relative tolerance for source-patch reconstruction.

    Returns:
        Validated bank-linked provenance, attached only after all patches pass.

    Raises:
        ValueError: Any source, metadata, feature state or reconstructed patch differs.

    Examples:
        >>> provenance = bind_reconstructed_reference_sources(  # doctest: +SKIP
        ...     model.eval(), saved_sources, source_batches,
        ...     source_context={"checkpoint_sha256": checkpoint_sha256})
    """
    _check_math(model)
    _require(
        not model.training and model.stage == "classifier",
        "Source reconstruction requires an eval-mode refitted model.",
    )
    _require(
        all(
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
            and v >= 0
            for v in (atol, rtol)
        ),
        "Invalid reconstruction tolerances.",
    )
    context = copy.deepcopy(source_context)
    _json_bytes(context)
    sources = copy.deepcopy(sources)
    branch = model.branches["morphology"]
    _require(len(sources) == branch.num_prototypes, "Source count differs.")
    by_id = {}
    for index, record in enumerate(sources):
        _require(record.get("prototype_id") == index, "Source prototype order differs.")
        eid = _integer(record["source_ecg_id"], "source_ecg_id")
        by_id.setdefault(eid, []).append((index, record))
    rows, seen, completed, max_error = [], set(), {}, 0.0
    bank_before = bank_state_digest(
        _live_tensors(model), model.label_names, model._architecture()
    )
    for batch in source_batches:
        x = batch[model.signal_key]
        y = batch[model.label_key]
        _require(
            isinstance(x, Tensor)
            and x.dtype == torch.float32
            and x.device.type == "cpu"
            and bool(torch.isfinite(x).all()),
            "Source inputs must be finite CPU float32 tensors.",
        )
        clean = model._signal(x)
        _require(
            y.shape == (len(clean), model.num_labels)
            and bool(((y == 0) | (y == 1)).all()),
            "Invalid source targets.",
        )
        for row in range(len(clean)):
            eid = _integer(batch["ecg_id"][row], "ecg_id")
            fold = _integer(batch["strat_fold"][row], "strat_fold")
            _require(
                eid in by_id and eid not in seen, "Unexpected or duplicate source ECG."
            )
            _require(fold in range(1, 9), "Source ECG is outside training folds.")
        _require(
            len(set(batch["ecg_id"])) == len(clean), "Duplicate source ECG in batch."
        )
        features = branch.feature_map(clean)
        for row in range(len(clean)):
            eid = _integer(batch["ecg_id"][row], "ecg_id")
            _require(
                eid in by_id and eid not in seen, "Unexpected or duplicate source ECG."
            )
            seen.add(eid)
            fold = _integer(batch["strat_fold"][row], "strat_fold")
            _require(fold in range(1, 9), "Source ECG is outside training folds.")
            patient = str(batch["patient_id"][row])
            _require(bool(patient), "Missing source patient ID.")
            target_labels = [
                model.label_names[i] for i in y[row].nonzero().flatten().tolist()
            ]
            signal_hash = _tensor_digest(clean[row])
            windows = features.shape[-1] - branch.window + 1
            for index, record in by_id[eid]:
                label = model.label_names[
                    int(branch.prototype_class_identity[index].argmax())
                ]
                _require(
                    record["assigned_label"] == label and label in target_labels,
                    "Source target does not carry the assigned label.",
                )
                start = _integer(record["latent_start"], "latent_start")
                _require(
                    start < windows
                    and record["latent_end_exclusive"] == start + branch.window,
                    "Source latent coordinate is invalid.",
                )
                patch = features[row, :, start : start + branch.window]
                prototype = branch.prototype_vectors[index]
                error = (patch - prototype).abs().max().item()
                max_error = max(max_error, error)
                _require(
                    torch.allclose(patch, prototype, atol=atol, rtol=rtol),
                    f"Source patch differs from prototype {index}: max error {error:g}",
                )
                score = float(record["pre_projection_similarity"])
                _require(math.isfinite(score), "Nonfinite saved projection similarity.")
                completed[index] = {
                    **record,
                    "sample": len(rows),
                    "window": start,
                    "similarity": score,
                    "source_patient_id": patient,
                    "source_fold": fold,
                    "source_labels": target_labels,
                    "source_signal_sha256": signal_hash,
                    "latent_num_windows": windows,
                    "prototype_vector_sha256": _tensor_digest(prototype),
                }
            rows.append(
                {
                    "ecg_id": eid,
                    "patient_id": patient,
                    "fold": fold,
                    "labels": target_labels,
                    "signal_sha256": signal_hash,
                }
            )
    _require(
        seen == set(by_id) and len(completed) == len(sources), "Missing source ECGs."
    )
    _require(
        bank_before
        == bank_state_digest(
            _live_tensors(model), model.label_names, model._architecture()
        ),
        "Feature/bank state changed.",
    )
    provenance = {
        "format_version": 1,
        "label_names": list(model.label_names),
        "architecture": model._architecture(),
        "source_records": [completed[i] for i in range(len(sources))],
        "covered_prototypes": len(sources),
        "input_records": len(rows),
        "input_manifest_sha256": hashlib.sha256(_json_bytes(rows)).hexdigest(),
        "source_context": {
            **context,
            "import_operation": "saved_patch_reconstruction_only",
            "patch_tolerances": {"atol": atol, "rtol": rtol},
            "patch_max_absolute_error": max_error,
        },
        "tie_break": "Historical saved selection; no search or tie resolution rerun",
        "bank_state_sha256": bank_before,
    }
    provenance["source_bundle_sha256"] = hashlib.sha256(
        _json_bytes(provenance)
    ).hexdigest()
    validate_projection_provenance(
        provenance, _live_tensors(model), model.label_names, model._architecture()
    )
    model._projection_provenance = provenance
    model.projection_info = {"morphology": copy.deepcopy(provenance["source_records"])}
    return copy.deepcopy(provenance)
