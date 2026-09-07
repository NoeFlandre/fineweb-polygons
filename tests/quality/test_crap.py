from scripts.check_crap import _blocks


def test_crap_blocks_score_functions_and_methods_not_class_containers() -> None:
    blocks = _blocks(
        """
class Example:
    def method(self):
        return 1

def function():
    return 2
"""
    )

    assert [block.fullname for block in blocks] == [
        "Example.method",
        "function",
    ]
