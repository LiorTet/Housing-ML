import pytest

from housing import data
from housing.features import FEATURES, TARGET, validate


def test_build_is_deterministic(settings):
    first, second = data.build(settings), data.build(settings)
    assert first.version == second.version
    assert set(first.stats) == {*FEATURES, TARGET}


def test_silver_rejects_contract_violation(settings, bronze):
    bad = bronze.assign(Latitude=bronze["Latitude"].where(bronze.index > 0, 99.0))
    _, manifest = data.bronze(settings)
    with pytest.raises(ValueError, match="Latitude"):
        data.silver(settings, bad, manifest)


def test_validate_reports_missing_columns(bronze):
    assert validate(bronze.drop(columns=["MedInc"])) == ["missing columns: ['MedInc']"]


def test_tampered_layer_fails(settings):
    gold = data.build(settings)
    stored = settings.data_dir / "gold" / gold.version / "data.csv"
    stored.write_text(stored.read_text().replace("train", "test", 1))
    with pytest.raises(ValueError, match="modified after commit"):
        data.load_gold(settings)
    with pytest.raises(ValueError, match="modified after commit"):
        data.build(settings)


def test_gold_lineage(settings):
    gold = data.build(settings)
    silver = data.Manifest.load(settings.data_dir / "silver" / gold.parent / "manifest.json")
    assert (settings.data_dir / "bronze" / silver.parent).exists()
