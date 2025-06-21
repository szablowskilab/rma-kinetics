edit notebook:
    @uv run marimo edit ./notebooks/{{notebook}}
docs-dev:
    uv run mkdocs serve
docs-build:
    uv run mkdocs build
app-dev:
    @uv run marimo edit ./app/main.py
app-serve:
    @uv run marimo run ./app/main.py
