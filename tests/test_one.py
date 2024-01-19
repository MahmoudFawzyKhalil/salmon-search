from salmon_search import db


def test_one():
    assert db.add(1, 2) == 3
