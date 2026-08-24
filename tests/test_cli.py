from fineweb_polygons.cli import main


def test_cli_reports_foundation_only(capsys) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out == (
        "fineweb-polygons foundation only; no pipeline executed\n"
    )
