"""Real Drosophila connectome loader (FAFB-FlyWire / Codex).

Public data hosted by the FlyWire consortium (Princeton NSI) under CC BY 4.0
at ``https://storage.googleapis.com/flywire-data/codex/data/fafb/783/``.

The tool ``python -m flybrain.data.fetch_flywire --all`` downloads the CSVs
into ``data/raw/flywire/``:

    data/raw/flywire/
    ├── neurons.csv.gz            root_id, group, nt_type, nt_type_score, *_avg
    ├── connections.csv.gz        pre_root_id, post_root_id, neuropil, syn_count, nt_type
    ├── classification.csv.gz     root_id, flow, super_class, class, sub_class,
    │                             hemilineage, side, nerve
    ├── consolidated_cell_types.csv.gz    root_id, primary_type, additional_type(s)
    └── coordinates.csv.gz        root_id, position (nm)

Schema note: the *neurons* file does NOT contain ``super_class``. That column
lives in *classification.csv.gz* — the loader merges them.

Scientific honesty (spec §3): the *structure* is real. The mapping of gesture
channels onto FlyWire cell types is documented in
:func:`_build_gesture_populations` and is still a computational convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import numpy as np

from flybrain.connectome.annotations import NeuronAnnotations
from flybrain.connectome.graph import Connectome, SynapticEdges


FLYWIRE_DIR = Path("data/raw/flywire")

# Real FlyWire ``super_class`` values (from classification.csv.gz).
SUPER_CLASS_VISUAL     = {"optic", "visual_projection", "visual_centrifugal"}
SUPER_CLASS_DESCENDING = {"descending"}
SUPER_CLASS_MOTOR      = {"motor"}
SUPER_CLASS_CENTRAL    = {"central"}
SUPER_CLASS_SENSORY    = {"sensory", "sensory_ascending"}


def load_flywire(
    data_dir: Path = FLYWIRE_DIR,
    include_super_classes: Optional[Iterable[str]] = None,
    min_syn_count: int = 5,
    max_neurons: Optional[int] = 20_000,
    map_gestures: bool = True,
) -> tuple[Connectome, NeuronAnnotations]:
    """Build a :class:`Connectome` from FlyWire CSVs.

    Parameters
    ----------
    data_dir
        Folder containing ``connections.csv[.gz]`` and ``neurons.csv[.gz]``.
    include_super_classes
        FlyWire ``super_class`` values to keep. Default:
        visual + central + descending — the pathway relevant to
        visual-motor experiments.
    min_syn_count
        Drop synaptic connections weaker than this many synapses (FlyWire's
        recommended threshold; noise below ~5 synapses).
    max_neurons
        Hard cap for laptop-friendly runs. If exceeded, keeps the highest
        in-degree neurons.
    map_gestures
        If ``True`` also register the 6 gesture sensory + 6 motor
        sub-populations (documented computational mapping — see below) so
        the existing dashboard / decoder work unchanged.

    Returns
    -------
    (connectome, annotations)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas required. `pip install -e '.[data]'`") from e

    data_dir = Path(data_dir)
    conn_csv = _first_existing(data_dir, ["connections.csv.gz", "connections.csv",
                                          "connections.feather"])
    neur_csv = _first_existing(data_dir, ["neurons.csv.gz", "neurons.csv",
                                          "neurons.feather"])
    class_csv = _first_existing(data_dir, ["classification.csv.gz", "classification.csv"])
    coord_csv = _first_existing(data_dir, ["coordinates.csv.gz", "coordinates.csv"])
    if conn_csv is None or neur_csv is None:
        raise FileNotFoundError(
            f"FlyWire data not found under {data_dir}.\n"
            "Run:  python -m flybrain.data.fetch_flywire --all\n"
            "or download from https://codex.flywire.ai/api/download"
        )

    print(f"[flywire] loading neurons from {neur_csv}")
    neurons = _read_table(neur_csv)
    if "root_id" not in neurons.columns:
        raise ValueError(f"neurons file missing 'root_id'. Got: {list(neurons.columns)}")

    # neurons.csv.gz has NO super_class — that lives in classification.csv.gz.
    # Merge it in so the rest of the loader has one dataframe.
    if class_csv is not None:
        print(f"[flywire] merging classification from {class_csv}")
        cls = _read_table(class_csv)
        merge_cols = [c for c in ["root_id", "flow", "super_class", "class",
                                  "sub_class", "side", "nerve", "hemilineage"]
                      if c in cls.columns]
        neurons = neurons.merge(cls[merge_cols], on="root_id", how="left")
    else:
        print("[flywire] ⚠️  classification.csv.gz missing — no cell-type labels")

    # Optional real neuron coordinates (nm) for the 3-D layout.
    # FlyWire ``coordinates.csv.gz`` stores position as a single string column
    # ``position`` of the form ``"[x y z]"`` in nanometers. Parse it into
    # numeric columns so we can use real anatomy in the visualiser.
    if coord_csv is not None:
        try:
            coords = _read_table(coord_csv)
            if "position" in coords.columns and "pos_x" not in coords.columns:
                # Format: "[352484 175164 229040]" (nm) — sometimes with commas
                # or extra spaces. Parse robustly.
                pos_str = (coords["position"].astype(str)
                           .str.strip()
                           .str.strip("[]")
                           .str.replace(",", " ", regex=False)
                           .str.strip())
                xyz = pos_str.str.extract(
                    r"(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)"
                ).astype(float)
                coords["pos_x"] = xyz[0].to_numpy(dtype=np.float32)
                coords["pos_y"] = xyz[1].to_numpy(dtype=np.float32)
                coords["pos_z"] = xyz[2].to_numpy(dtype=np.float32)
                coords = coords.drop(columns=["position"])
            # Some neurons have multiple soma-position rows — keep the first.
            coords = coords.drop_duplicates(subset="root_id", keep="first")
            neurons = neurons.merge(
                coords[[c for c in ("root_id", "pos_x", "pos_y", "pos_z")
                        if c in coords.columns]],
                on="root_id", how="left")
            print(f"[flywire] merged real 3-D coordinates from {coord_csv}")
        except Exception as e:
            print(f"[flywire] coordinates merge failed: {e}")

    include = set(include_super_classes) if include_super_classes else \
        (SUPER_CLASS_VISUAL | SUPER_CLASS_CENTRAL |
         SUPER_CLASS_DESCENDING | SUPER_CLASS_MOTOR)
    if "super_class" in neurons.columns:
        before = len(neurons)
        neurons = neurons[neurons["super_class"].isin(include)].reset_index(drop=True)
        print(f"[flywire]   kept {len(neurons):,}/{before:,} neurons "
              f"(super_class ∈ {sorted(include)})")

    print(f"[flywire] loading connections from {conn_csv}")
    edges = _read_table(conn_csv)
    for col in ("pre_root_id", "post_root_id", "syn_count"):
        if col not in edges.columns:
            raise ValueError(f"connections file missing '{col}'. Got: {list(edges.columns)}")
    edges = edges[edges["syn_count"] >= min_syn_count]
    keep_ids = set(neurons["root_id"].to_numpy().tolist())
    edges = edges[edges["pre_root_id"].isin(keep_ids) &
                  edges["post_root_id"].isin(keep_ids)]
    print(f"[flywire]   kept {len(edges):,} edges (syn_count≥{min_syn_count})")

    # Laptop cap — keep the highest in-degree neurons + always keep all descending/motor
    if max_neurons is not None and len(neurons) > max_neurons:
        indeg = edges.groupby("post_root_id")["syn_count"].sum()
        # Force-include descending + motor + T4/T5 motion detectors so key
        # populations survive the cap.
        force_keep = set()
        if "super_class" in neurons.columns:
            force_keep |= set(neurons.loc[
                neurons["super_class"].isin(SUPER_CLASS_DESCENDING | SUPER_CLASS_MOTOR),
                "root_id"].tolist())
        if "sub_class" in neurons.columns:
            force_keep |= set(neurons.loc[
                neurons["sub_class"].isin(["t4_neuron", "t5_neuron"]),
                "root_id"].tolist())
        top = list(indeg.sort_values(ascending=False).head(max_neurons).index)
        top_set = set(top) | force_keep
        neurons = neurons[neurons["root_id"].isin(top_set)].reset_index(drop=True)
        keep_ids = set(neurons["root_id"].tolist())
        edges = edges[edges["pre_root_id"].isin(keep_ids) &
                      edges["post_root_id"].isin(keep_ids)]
        print(f"[flywire]   capped to top-{max_neurons} + all descending/motor "
              f"({len(neurons):,} neurons, {len(edges):,} edges)")

    # Reindex root_id → contiguous [0, N)
    ids = neurons["root_id"].to_numpy()
    id_to_ix = {int(v): i for i, v in enumerate(ids)}
    pre = edges["pre_root_id"].map(id_to_ix).to_numpy(dtype=np.int32)
    post = edges["post_root_id"].map(id_to_ix).to_numpy(dtype=np.int32)
    weights = edges["syn_count"].to_numpy(dtype=np.float32)

    conn = Connectome(
        num_neurons=len(ids),
        edges=SynapticEdges(pre=pre, post=post, weights=weights),
        neuron_ids=ids.astype(np.int64),
    )

    ann = NeuronAnnotations(num_neurons=conn.num_neurons)
    if "super_class" in neurons.columns:
        for sc in neurons["super_class"].dropna().unique():
            idx = np.where(neurons["super_class"].to_numpy() == sc)[0]
            ann.add_population(f"super_{sc}", idx.astype(np.int32))
    if "class" in neurons.columns:
        ann.cell_type = neurons["class"].fillna("").to_numpy()
        # -- Biological *brain regions* (matches FlyWire ``class`` labels).
        # These are the populations the dashboard visualises with the coloured
        # anatomical regions in the mockup: Optic Lobe, Central Complex,
        # Mushroom Body, Antennal Lobe, Motor Circuits.
        sc = neurons["super_class"].to_numpy() if "super_class" in neurons.columns \
             else np.array([""] * len(neurons))
        cls = neurons["class"].fillna("").to_numpy()
        region_masks = {
            "region_optic_lobe":     np.isin(sc, list(SUPER_CLASS_VISUAL)),
            "region_central_complex": np.isin(cls, ["CX"]),
            "region_mushroom_body":   np.isin(cls, ["Kenyon_Cell", "MBON", "MBIN", "DAN"]),
            "region_antennal_lobe":   np.isin(cls, ["olfactory", "ALPN", "ALLN", "ALIN",
                                                     "ALON", "mAL"]),
            "region_motor":           np.isin(sc, list(SUPER_CLASS_DESCENDING | SUPER_CLASS_MOTOR)),
        }
        for name, mask in region_masks.items():
            idx = np.where(mask)[0]
            if idx.size:
                ann.add_population(name, idx.astype(np.int32))
        # "interneuron" = everything else in central brain not in the named regions
        assigned = np.zeros(len(neurons), dtype=bool)
        for mask in region_masks.values():
            assigned |= mask
        inter = np.isin(sc, list(SUPER_CLASS_CENTRAL)) & ~assigned
        if inter.any():
            ann.add_population("region_interneuron",
                               np.where(inter)[0].astype(np.int32))
        print(f"[flywire]   regions: "
              + ", ".join(f"{k.replace('region_',''):>16}={m.sum()}" for k,m in region_masks.items()))
    if "sub_class" in neurons.columns:
        # add sub-class populations for T4/T5 etc. — useful for future analyses
        for st in ["t4_neuron", "t5_neuron"]:
            idx = np.where(neurons["sub_class"].to_numpy() == st)[0]
            if idx.size:
                ann.add_population(f"sub_{st}", idx.astype(np.int32))
    # Real 3-D coordinates (nm) if present
    for xyz_col in [("pos_x", "pos_y", "pos_z"),
                    ("position_x", "position_y", "position_z"),
                    ("x", "y", "z")]:
        if all(c in neurons.columns for c in xyz_col):
            ann.xyz = neurons[list(xyz_col)].to_numpy(dtype=np.float32)
            break

    if map_gestures:
        _build_gesture_populations(ann, neurons)

    print(f"[flywire] ✅ connectome ready: N={conn.num_neurons:,}  edges={conn.W.nnz:,}")
    return conn, ann


# ------------------------------------------------------------- gesture mapping
def _build_gesture_populations(ann: NeuronAnnotations, neurons) -> None:
    """Attach the 6 sensory + 6 motor gesture sub-populations.

    Uses real biology where FlyWire annotations allow:

    * **T4 / T5 neurons** are Drosophila's direction-selective motion
      detectors (Maisak et al. 2013). If ``class == 't4_neuron'`` /
      ``'t5_neuron'`` labels are present, we split them by ``side``
      (left/right) as an approximate mapping onto LEFT/RIGHT/UP/DOWN
      motion channels. This is a documented **approximation** — T4/T5
      cells actually encode four cardinal motion directions each, but the
      per-cell sub-direction label isn't in the public release schema.
    * **Descending neurons** carry motor commands to the ventral nerve
      cord — biologically the correct "motor output" pool. We split them
      by ``side`` for left/right rolls/yaws.

    Any split within these that isn't backed by annotations is labelled
    as computational convention in the code and docstring.
    """
    import pandas as pd  # noqa: F401
    if "super_class" not in neurons.columns:
        return
    sc = neurons["super_class"].to_numpy()
    sub = neurons["sub_class"].to_numpy() if "sub_class" in neurons.columns \
        else np.array([""] * len(neurons))
    side = neurons["side"].fillna("center").to_numpy() if "side" in neurons.columns else \
        np.array(["center"] * len(neurons))

    visual_mask = np.isin(sc, list(SUPER_CLASS_VISUAL))
    desc_mask   = np.isin(sc, list(SUPER_CLASS_DESCENDING))
    motor_mask  = np.isin(sc, list(SUPER_CLASS_MOTOR))
    central_mask = np.isin(sc, list(SUPER_CLASS_CENTRAL))

    if central_mask.any():
        ann.add_population("interneuron", np.where(central_mask)[0].astype(np.int32))

    # -------- sensory: prefer real T4/T5 motion detectors
    t4 = np.where(visual_mask & (sub == "t4_neuron"))[0]
    t5 = np.where(visual_mask & (sub == "t5_neuron"))[0]
    if t4.size + t5.size > 100:
        # T4 = ON-edge motion, T5 = OFF-edge motion.
        # Split each by side, then further into halves for finer direction hints.
        t4_l = t4[side[t4] == "left"];  t4_r = t4[side[t4] == "right"]
        t5_l = t5[side[t5] == "left"];  t5_r = t5[side[t5] == "right"]
        # Fallback if a side is empty
        def _half(a): return np.array_split(a, 2)
        up,   down   = _half(t4_l) if t4_l.size else (t4[:len(t4)//2], t4[len(t4)//2:])
        left, right  = _half(t5_l) if t5_l.size else (t5[:len(t5)//2], t5[len(t5)//2:])
        rot_cw, rot_ccw = (t4_r, t5_r) if (t4_r.size and t5_r.size) else \
                          _half(np.where(visual_mask)[0])
        ann.add_population("sensory_up",      up.astype(np.int32))
        ann.add_population("sensory_down",    down.astype(np.int32))
        ann.add_population("sensory_left",    left.astype(np.int32))
        ann.add_population("sensory_right",   right.astype(np.int32))
        ann.add_population("sensory_rot_cw",  np.asarray(rot_cw, dtype=np.int32))
        ann.add_population("sensory_rot_ccw", np.asarray(rot_ccw, dtype=np.int32))
        print(f"[flywire]   sensory: T4={t4.size} T5={t5.size} "
              f"→ up/down/left/right/rot_cw/rot_ccw = "
              f"{up.size}/{down.size}/{left.size}/{right.size}/"
              f"{np.asarray(rot_cw).size}/{np.asarray(rot_ccw).size}")
    else:
        # Fallback: arbitrary 6-way split over visual super_classes
        visual_idx = np.where(visual_mask)[0]
        names = ["sensory_up", "sensory_down", "sensory_left",
                 "sensory_right", "sensory_rot_cw", "sensory_rot_ccw"]
        for name, part in zip(names, np.array_split(visual_idx, 6)):
            ann.add_population(name, part.astype(np.int32))
        print(f"[flywire]   sensory: T4/T5 not found — arbitrary 6-way split "
              f"of {visual_idx.size:,} visual neurons")

    # -------- motor: descending (+ real motor) neurons, split by side
    output_pool = np.where(desc_mask | motor_mask)[0]
    if output_pool.size == 0:
        return
    side_out = side[output_pool]
    left_out  = output_pool[side_out == "left"]
    right_out = output_pool[side_out == "right"]
    if left_out.size < 3 or right_out.size < 3:
        left_out, right_out = np.array_split(output_pool, 2)
    # Split each side into 3 sub-populations
    ann.add_population("motor_thrust",   np.array_split(left_out, 3)[0].astype(np.int32))
    ann.add_population("motor_descent",  np.array_split(left_out, 3)[1].astype(np.int32))
    ann.add_population("motor_roll_l",   np.array_split(left_out, 3)[2].astype(np.int32))
    ann.add_population("motor_roll_r",   np.array_split(right_out, 3)[0].astype(np.int32))
    ann.add_population("motor_yaw_cw",   np.array_split(right_out, 3)[1].astype(np.int32))
    ann.add_population("motor_yaw_ccw",  np.array_split(right_out, 3)[2].astype(np.int32))
    if np.where(visual_mask)[0].size:
        ann.add_population("visual", np.where(visual_mask)[0].astype(np.int32))
    ann.add_population("motor", output_pool.astype(np.int32))
    print(f"[flywire]   motor: {output_pool.size} descending+motor "
          f"(L={left_out.size}, R={right_out.size})")


# ------------------------------------------------------------------ utilities
def _first_existing(root: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _read_table(path: Path):
    import pandas as pd
    if path.suffix == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)
