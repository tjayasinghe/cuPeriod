"""Partitions: builders, edits, origins and the suggested split (E2)."""

from __future__ import annotations

import numpy as np
import pydantic
import pytest

from cuperiod.prewhiten.subsets import (
    ChunksOrigin,
    CrossedOrigin,
    GapsOrigin,
    LabelsOrigin,
    ManualOrigin,
    Partition,
    RangesOrigin,
    parse_origin,
    suggest_partition,
)


def _nights(n_nights: int = 5, per_night: int = 20, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [k + rng.uniform(0.0, 0.3, per_night) for k in range(n_nights)]
    ) + 2459000.0


# --- builders ----------------------------------------------------------------------


def test_from_gaps_finds_nights_in_any_row_order() -> None:
    t = _nights()
    rng = np.random.default_rng(2)
    shuffled = rng.permutation(t)
    part = Partition.from_gaps(shuffled, min_gap=0.3)
    assert part.n_subsets == 5
    assert part.labels == ("N1", "N2", "N3", "N4", "N5")
    assert part.counts().tolist() == [20] * 5
    # Codes follow time order: N1 is the earliest night.
    spans = part.spans(shuffled)
    assert np.all(np.diff(spans[:, 0]) > 0)
    assert np.array_equal(part.codes, np.floor(shuffled - 2459000.0).astype(int))
    assert part.origin == GapsOrigin(min_gap=0.3)
    assert part.codes.dtype == np.int32
    assert not part.codes.flags.writeable


def test_from_gaps_merges_small_groups_across_the_smaller_gap() -> None:
    # Nights at 0, 1, 1.4 (a 5-point stub), 3: the stub is 0.1 d after night 1 ends
    # and 1.3 d before night 3 starts, so it joins night 1.
    t = np.concatenate([
        np.linspace(0.0, 0.3, 20),
        np.linspace(1.0, 1.3, 20),
        np.linspace(1.4, 1.45, 5),
        np.linspace(3.0, 3.3, 20),
    ])
    part = Partition.from_gaps(t, min_gap=0.05, min_points=10)
    assert part.counts().tolist() == [20, 25, 20]
    assert part.codes[40:45].tolist() == [1] * 5


def test_from_gaps_with_max_span_finds_sectors() -> None:
    # Two TESS-like sectors, each with a 1-day mid-sector downlink gap.
    pieces = [
        np.arange(0.0, 13.0, 0.02), np.arange(14.0, 27.0, 0.02),
        np.arange(28.0, 41.0, 0.02), np.arange(42.0, 55.0, 0.02),
    ]
    t = np.concatenate(pieces)
    assert Partition.from_gaps(t, min_gap=0.5).n_subsets == 4
    sectors = Partition.from_gaps(t, min_gap=0.5, max_span=28.5, name="sector",
                                  prefix="S")
    assert sectors.n_subsets == 2 and sectors.labels == ("S1", "S2")
    assert sectors.counts().tolist() == [pieces[0].size + pieces[1].size,
                                         pieces[2].size + pieces[3].size]


def test_from_chunks() -> None:
    t = np.arange(0.0, 30.0, 0.1)
    part = Partition.from_chunks(t, length=10.0)
    assert part.counts().tolist() == [100, 100, 100]
    assert part.labels == ("C1", "C2", "C3")
    shifted = Partition.from_chunks(t, length=10.0, origin=-5.0, min_points=1)
    assert shifted.counts().tolist() == [50, 100, 100, 50]
    # A 3-point tail chunk is merged into its neighbour.
    tail = np.r_[np.arange(0.0, 20.0, 0.1), [25.0, 25.1, 25.2]]
    assert Partition.from_chunks(tail, length=10.0).counts().tolist() == [100, 103]
    assert part.origin == ChunksOrigin(length=10.0)


def test_from_labels() -> None:
    values = np.array(["V", "B", "V", "R", "B"])
    part = Partition.from_labels(values, name="band")
    assert part.labels == ("B", "R", "V")
    assert part.codes.tolist() == [2, 0, 2, 1, 0]
    assert part.origin == LabelsOrigin(column="band")
    numeric = Partition.from_labels(np.array([10, 2, 2]), name="site")
    assert numeric.labels == ("2", "10")


def test_from_ranges() -> None:
    t = np.arange(10.0)
    part = Partition.from_ranges(t, [3.0, 7.0], labels=["early", "mid", "late"])
    assert part.labels == ("early", "mid", "late")
    assert part.codes.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    # A range with no point is dropped with its label.
    sparse = Partition.from_ranges(t, [3.2, 3.5, 7.0])
    assert sparse.labels == ("R1", "R3", "R4")
    with pytest.raises(ValueError, match="3 ranges"):
        Partition.from_ranges(t, [3.0, 7.0], labels=["a", "b"])
    with pytest.raises(pydantic.ValidationError):
        Partition.from_ranges(t, [7.0, 3.0])


def test_from_runs_round_trip() -> None:
    part = Partition.from_gaps(_nights(), min_gap=0.3).with_weights([1, 2, 3, 4, 5])
    rebuilt = Partition.from_runs(
        part.n_points, part.runs(), name=part.name, labels=part.labels,
        weights=part.weights,
    )
    assert np.array_equal(rebuilt.codes, part.codes)
    assert rebuilt.labels == part.labels and rebuilt.weights == part.weights
    assert rebuilt.origin == ManualOrigin()


@pytest.mark.parametrize(
    "runs",
    [
        [(0, 5, 0), (6, 10, 1)],
        [(0, 5, 0), (5, 9, 1)],
        [(0, 5, 0), (5, 11, 1)],
        [(1, 10, 0)],
        [(0, 5, 0, 1)],
    ],
)
def test_from_runs_rejects_bad_tilings(runs) -> None:
    with pytest.raises(ValueError):
        Partition.from_runs(10, runs, name="x", labels=("a", "b"))


@pytest.mark.parametrize(
    ("origin", "kwargs"),
    [
        (GapsOrigin(min_gap=0.3), {}),
        (ChunksOrigin(length=2.0, min_points=5), {}),
        (LabelsOrigin(column="site"), {"labels": {"site": None}}),
        (RangesOrigin(edges=[2459001.5, 2459003.5]), {}),
    ],
)
def test_from_origin_reproduces_the_builder(origin, kwargs) -> None:
    t = _nights()
    if "labels" in kwargs:
        kwargs = {"labels": {"site": np.where(t < 2459002.0, "A", "B")}}
    part = Partition.from_origin(origin, t, name="p", **kwargs)
    assert part.origin == origin
    again = Partition.from_origin(part.origin, t, name="p", **kwargs)
    assert np.array_equal(again.codes, part.codes) and again.labels == part.labels


def test_from_origin_crossed_and_manual() -> None:
    t = _nights()
    night = Partition.from_gaps(t, min_gap=0.3)
    site = Partition.from_labels(np.where(np.arange(t.size) % 2 == 0, "A", "B"),
                                 name="site")
    crossed = Partition.from_origin(
        CrossedOrigin(a="site", b="night"), t, name="sxn",
        partitions={"site": site, "night": night},
    )
    assert crossed.n_subsets == 10
    with pytest.raises(ValueError, match="crossing needs"):
        Partition.from_origin(CrossedOrigin(a="site", b="x"), t, name="y",
                              partitions={"site": site})
    with pytest.raises(ValueError, match="from_runs"):
        Partition.from_origin(ManualOrigin(), t, name="m")
    with pytest.raises(ValueError, match="no label column"):
        Partition.from_origin(LabelsOrigin(column="site"), t, name="s")


# --- edits ----------------------------------------------------------------------------


def test_crossed() -> None:
    site = Partition(name="site", codes=np.array([0, 0, 1, 1]), labels=("A", "B"),
                     weights=(1.0, 2.0), origin=ManualOrigin())
    night = Partition(name="night", codes=np.array([0, 1, 0, 0]), labels=("N1", "N2"),
                      weights=(3.0, 1.0), origin=ManualOrigin())
    both = site.crossed(night)
    assert both.labels == ("A/N1", "A/N2", "B/N1")
    assert both.codes.tolist() == [0, 1, 2, 2]
    assert both.weights == (3.0, 1.0, 6.0)
    assert both.name == "sitexnight"
    assert both.origin == CrossedOrigin(a="site", b="night")


def test_merged() -> None:
    part = Partition.from_gaps(_nights(), min_gap=0.3).with_weights([1, 2, 3, 4, 5])
    merged = part.merged(3, 1)
    assert merged.labels == ("N1", "N2+N4", "N3", "N5")
    assert merged.weights == (1.0, 2.0, 3.0, 5.0)
    assert merged.counts().tolist() == [20, 40, 20, 20]
    assert merged.origin == ManualOrigin()
    with pytest.raises(ValueError):
        part.merged(2, 2)
    with pytest.raises(ValueError):
        part.merged(0, 9)


def test_merged_small() -> None:
    t = np.concatenate([np.linspace(0, 0.2, 20), np.linspace(1, 1.2, 4),
                        np.linspace(2, 2.2, 20), np.linspace(2.4, 2.45, 3)])
    part = Partition.from_gaps(t, min_gap=0.1, min_points=1)
    assert part.counts().tolist() == [20, 4, 20, 3]
    fixed = part.merged_small(t, min_points=10)
    assert fixed.counts().tolist() == [24, 23]
    assert fixed.labels == ("N1+N2", "N3+N4")
    assert fixed.origin == ManualOrigin()


def test_split_at() -> None:
    t = np.arange(20.0)
    part = Partition.from_chunks(t, length=10.0, min_points=1)
    split = part.split_at(5.0, t)
    assert split.labels == ("C1a", "C1b", "C2")
    assert split.counts().tolist() == [5, 5, 10]
    assert split.origin == ManualOrigin()
    with pytest.raises(ValueError, match="both sides"):
        part.split_at(10.0, t)


def test_with_weights_and_renamed() -> None:
    part = Partition.from_chunks(np.arange(20.0), length=10.0, min_points=1)
    weighted = part.with_weights([1.0, 0.25])
    assert weighted.weights == (1.0, 0.25)
    with pytest.raises(ValueError):
        part.with_weights([1.0])
    with pytest.raises(ValueError, match="finite and positive"):
        part.with_weights([1.0, 0.0])
    renamed = part.renamed(1, "late")
    assert renamed.labels == ("C1", "late")
    with pytest.raises(ValueError, match="duplicate"):
        part.renamed(1, "C1")
    with pytest.raises(ValueError):
        part.renamed(1, "")


def test_counts_and_spans_with_a_mask() -> None:
    t = np.arange(20.0)
    part = Partition.from_chunks(t, length=10.0, min_points=1)
    used = np.ones(20, dtype=bool)
    used[10:] = False
    assert part.counts(used).tolist() == [10, 0]
    spans = part.spans(t, used)
    assert spans[0].tolist() == [0.0, 9.0]
    assert np.all(np.isnan(spans[1]))


def test_partition_validation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        Partition(name="p", codes=np.array([0, 2]), labels=("a", "b"),
                  weights=(1.0, 1.0), origin=ManualOrigin())
    with pytest.raises(ValueError, match="one weight per subset"):
        Partition(name="p", codes=np.array([0, 1]), labels=("a", "b"),
                  weights=(1.0,), origin=ManualOrigin())
    with pytest.raises(ValueError, match="integer"):
        Partition(name="p", codes=np.array([0.0, 1.0]), labels=("a", "b"),
                  weights=(1.0, 1.0), origin=ManualOrigin())
    with pytest.raises(ValueError, match="name"):
        Partition(name="", codes=np.array([0]), labels=("a",), weights=(1.0,),
                  origin=ManualOrigin())


# --- origins ---------------------------------------------------------------------


def test_origins_are_frozen_and_strict() -> None:
    origin = GapsOrigin(min_gap=0.3)
    with pytest.raises(pydantic.ValidationError):
        origin.min_gap = 1.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        GapsOrigin(min_gap=0.0)
    with pytest.raises(pydantic.ValidationError):
        GapsOrigin(min_gap=0.3, prefix="TOOLONGPREFIX")
    with pytest.raises(pydantic.ValidationError):
        parse_origin({"kind": "gaps", "min_gap": 0.3, "surprise": 1})
    with pytest.raises(pydantic.ValidationError):
        parse_origin({"kind": "pickle"})
    with pytest.raises(pydantic.ValidationError):
        RangesOrigin(edges=[0.0] * 1001)


@pytest.mark.parametrize(
    "origin",
    [
        GapsOrigin(min_gap=0.3, max_span=28.5, min_points=5, prefix="S"),
        ChunksOrigin(length=10.0, origin=2459000.0),
        LabelsOrigin(column="site"),
        RangesOrigin(edges=[1.0, 2.0], labels=["a", "b", "c"]),
        CrossedOrigin(a="site", b="night"),
        ManualOrigin(),
    ],
)
def test_origins_round_trip_through_json(origin) -> None:
    assert parse_origin(origin.model_dump(mode="json")) == origin


# --- suggestion ---------------------------------------------------------------------


def test_suggest_partition() -> None:
    assert suggest_partition(_nights()) == GapsOrigin(min_gap=0.3)
    continuous = np.arange(0.0, 27.0, 0.01)
    assert suggest_partition(continuous) is None
    seasons = np.r_[np.arange(0.0, 50.0, 0.05), np.arange(400.0, 450.0, 0.05)]
    assert suggest_partition(seasons) == GapsOrigin(min_gap=60.0, prefix="S")
    sectors = np.r_[np.arange(0.0, 13.0, 0.02), np.arange(14.0, 27.0, 0.02),
                    np.arange(28.0, 41.0, 0.02), np.arange(42.0, 55.0, 0.02)]
    assert suggest_partition(sectors) == GapsOrigin(
        min_gap=0.5, max_span=28.5, prefix="S"
    )
    assert suggest_partition(np.array([1.0])) is None
