"""Alembic migration environment, shipped as a real Python package (P5.1).

Why this file exists
--------------------
``pyproject.toml`` declares::

    [tool.setuptools.packages.find]
    include = ["app*"]

which only discovers *packages* -- directories containing ``__init__.py``. Without
this marker ``app/db/migrations/`` (alembic.ini, env.py, script.py.mako) and
``app/db/migrations/versions/`` are invisible to setuptools, so an installed
distribution silently ships no migrations at all and
``alembic -c app/db/migrations/alembic.ini upgrade head`` fails with a missing
config file.

The marker alone does **not** make the containerised path work: alembic.ini
declares ``script_location = app/db/migrations``, which resolves against the
*current working directory*, not against site-packages. That is why the Dockerfile
runs the app from the source tree (WORKDIR /srv/app) rather than from an installed
wheel. Both fixes are applied: this file makes a built wheel complete, and the
source tree makes the image work.

Alembic still loads ``env.py`` and each revision file *by path*, so turning these
directories into packages changes nothing about how migrations execute.
"""