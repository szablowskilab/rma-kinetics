edit notebook:
    @uv run marimo edit ./notebooks/{{notebook}}
docs-dev:
    uv run mkdocs serve
docs-build:
    uv run mkdocs build
