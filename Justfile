edit notebook:
    @uv run marimo edit ./notebooks/{{notebook}}
docs-dev:
    uv run --group docs mkdocs serve
docs-build:
    uv run mkdocs --group build
app-dev:
    @uv run marimo edit ./app/main.py
app-serve:
    @uv run marimo run ./app/main.py
