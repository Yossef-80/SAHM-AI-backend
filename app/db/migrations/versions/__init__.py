"""Alembic revision scripts, shipped as a real Python package (P5.1).

Alembic discovers revisions by scanning this directory for files, so it does not
need a package marker to find them. The marker exists purely so
``[tool.setuptools.packages.find] include = ["app*"]`` in ``pyproject.toml``
includes the revision scripts in a built distribution -- without it a wheel
contains the application and none of its migrations.

See ``app/db/migrations/__init__.py`` for the full explanation.
"""