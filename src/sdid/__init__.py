from .model import SyntheticDID
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("SyntheticDID")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["SyntheticDID"]