import json

import pytest

from northvale import config as C


@pytest.fixture(scope="session")
def m():
    if not C.METRICS.exists():
        pytest.skip("reports/metrics.json not built yet (run make analysis)")
    return json.loads(C.METRICS.read_text())


@pytest.fixture(scope="session")
def loans_path():
    if not C.LOANS.exists():
        pytest.skip("data/interim not built (run make data build)")
    return C.LOANS
