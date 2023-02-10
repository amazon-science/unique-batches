import importlib
import os

__MODULE_PREFIX__ = "unique_batches.data.datasets."

# Iterate all files in the same directory
for file in os.listdir(os.path.dirname(__file__)):
    # Exclude __init__.py and other non-python files
    if file.endswith(".py") and not file.startswith("_"):
        # Remove the .py extension
        module_name = file[: -len(".py")]
        importlib.import_module(__MODULE_PREFIX__ + module_name)
