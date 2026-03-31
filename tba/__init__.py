"""TBA — Thermal Budget Annealing for ML deployment optimization."""


def __getattr__(name):
    if name == "optimize_deployment":
        from tba.api import optimize_deployment
        return optimize_deployment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["optimize_deployment"]
