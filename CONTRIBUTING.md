# Contributing

Thanks for considering a contribution to `stereo_matching`. For the model
integration pattern, see [docs/adding_a_model.md](docs/adding_a_model.md).

## Setup

```bash
git clone https://github.com/shriarul5273/stereo_matching.git
cd stereo_matching
pip install -e ".[dev]"
```

## Running checks

Run the checks used for pull requests:

```bash
ruff check .
mypy src/stereo_matching
pytest tests -m "not slow"
python -m build
twine check dist/*
```

Ruff, the build, and the fast tests gate CI. Mypy is advisory while the
existing annotation backlog is reduced.

The development extra includes ONNX and ONNX Runtime. Export or quantization
changes should cover the two-input graph, ONNX Runtime verification, and both
the Python API and CLI where applicable. Users who only need deployment tools
can install the smaller `export` extra with `pip install -e ".[export]"`.

The repository's documentation must describe implemented behavior. Dataset,
evaluation, trainer, and loss APIs that are not present in `src/stereo_matching`
should be labeled as reserved or shown as application-owned examples rather
than documented as importable package features.

The slow tier downloads real pretrained weights and runs every registered
variant. It is scheduled weekly and can also be started manually:

```bash
pytest tests -m slow
```

Expect this tier to use substantial bandwidth, disk space, memory, and time.
Run it locally when changing checkpoint loading, weight mapping, preprocessing,
or model forward paths.

## Code style

- Match the conventions used by the neighboring model packages.
- Keep changes focused and avoid unrelated rewrites.
- Explain non-obvious constraints and workarounds, not straightforward code.
- Verify behavioral claims in docstrings and documentation.

## Pull requests

- Include the checks you ran and their results in the description.
- Add or update tests for behavior changes.
- Update every affected README or `docs/*.md` page and verify relative links.
- Keep generated artifacts and model checkpoints out of commits.
- Ensure lint, build, and the Python 3.10–3.12 fast-test matrix pass.

## Reporting issues

Open an issue at
[github.com/shriarul5273/stereo_matching/issues](https://github.com/shriarul5273/stereo_matching/issues).
For bugs, include the model variant, Python and torch versions, device, input
shape, and a minimal reproduction when possible.

## License

By contributing, you agree that your contributions are licensed under this
project's [MIT License](LICENSE).
