from brain_1.serving.api import build_app


def test_build_app() -> None:
    app = build_app()
    assert app.title == "brain-1 API"
