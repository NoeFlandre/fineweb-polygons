from scripts.check_mutation import _bad_mutation_lines


def test_mutation_gate_rejects_segfault_status() -> None:
    output = "fineweb_polygons.example.mutant: segfault\n"

    assert _bad_mutation_lines(output) == ["fineweb_polygons.example.mutant: segfault"]


def test_mutation_gate_ignores_killed_status() -> None:
    output = "fineweb_polygons.example.mutant: killed\n"

    assert _bad_mutation_lines(output) == []
