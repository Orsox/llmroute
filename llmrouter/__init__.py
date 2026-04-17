"""LM Router package entry points."""

__all__ = ["app", "create_app", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .app import app, create_app, main

        mapping = {
            "app": app,
            "create_app": create_app,
            "main": main,
        }
        return mapping[name]
    raise AttributeError(name)
