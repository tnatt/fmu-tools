# -*- coding: utf-8 -*-
"""Test code for RMS volumetrics parsing"""
from os.path import abspath
import pytest
from fmu.tools.qcproperties.qcproperties import QCProperties

PATH = abspath("tests/data/propstatistics")
WELLS = ["OP_1.bw"]
PROPERTIES = {
    "PORO": {"name": "Poro"},
    "PERM": {"name": "Perm"},
}
SELECTORS = {
    "ZONE": {"name": "Zonelog"},
    "FACIES": {"name": "Facies"},
}

qcp = QCProperties()


def test_full_dataframe():
    data = {
        "path": PATH,
        "wells": WELLS,
        "properties": PROPERTIES,
        "selectors": SELECTORS,
    }

    stat = qcp.get_bwell_statistics(data)

    assert set(stat.property_dataframe.columns) == set(
        ["ZONE", "PERM", "PORO", "FACIES"]
    )
    assert stat.property_dataframe["PORO"].mean() == pytest.approx(0.1673, abs=0.001)


def test_filters():
    data = {
        "path": PATH,
        "wells": WELLS,
        "properties": PROPERTIES,
        "selectors": {
            "ZONE": {"name": "Zonelog", "exclude": "TopLowerReek_BaseLowerReek"},
            "FACIES": {"name": "Facies", "include": "Channel"},
        },
    }

    stat = qcp.get_bwell_statistics(data)

    assert set(["Channel", "Total"]) == set(stat.dataframe["FACIES"].unique())
    assert set(stat.dataframe["ZONE"].unique()) == set(
        ["TopMidReek_TopLowerReek", "TopUpperReek_TopMidReek", "Total"]
    )


def test_statistics():

    data = {
        "path": PATH,
        "wells": WELLS,
        "properties": PROPERTIES,
        "selectors": SELECTORS,
        "name": "Blocked_Logs",
    }
    stat = qcp.get_bwell_statistics(data)

    assert set(stat.dataframe.columns) == set(
        [
            "Avg",
            "FACIES",
            "Max",
            "Min",
            "P10",
            "P90",
            "PROPERTY",
            "Stddev",
            "ZONE",
            "SOURCE",
            "ID",
        ]
    )
    assert list(stat.dataframe["ID"].unique())[0] == "Blocked_Logs"
    assert set(stat.dataframe["PROPERTY"].unique()) == set(["PERM", "PORO"])
    assert stat.dataframe[stat.dataframe["PROPERTY"] == "PORO"][
        "Avg"
    ].max() == pytest.approx(0.2913, abs=0.001)
    assert set(stat.dataframe["ZONE"].unique()) == set(
        [
            "TopLowerReek_BaseLowerReek",
            "TopMidReek_TopLowerReek",
            "TopUpperReek_TopMidReek",
            "Total",
        ]
    )

    row = stat.dataframe[
        (stat.dataframe["ZONE"] == "Total")
        & (stat.dataframe["FACIES"] == "Total")
        & (stat.dataframe["PROPERTY"] == "PORO")
    ]
    assert row["Avg"].values == pytest.approx(0.1673, abs=0.001)
