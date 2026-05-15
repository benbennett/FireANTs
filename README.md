# :fire: FireANTs (Windows Fork)

Fork of [rohitrango/fireants](https://github.com/rohitrango/fireants)
with native Windows support -- no Docker or WSL needed.

The FireANTs library is a lightweight, general-purpose registration
package for Riemannian diffeomorphic registration on GPUs. It is
designed to be easy to use, fast, accurate, and extensible.

See the [original repo](https://github.com/rohitrango/fireants) for
full documentation, tutorials, datasets, and citations.

## Differences from upstream

- Runs natively on Windows -- no Docker or WSL needed
- `pip install -e .` installs `fireantsRegistration` as a console entry point
- Debug/Release CUDA build flags via `python setup.py build_ext --debug`
- Cross-platform file locking (portalocker) and distributed backend (gloo)
- Python test runner: `python run_tests.py`

## Installation

```
git clone https://github.com/benbennett/FireANTs
cd FireANTs
pip install -e .
cd fused_ops && python setup.py build_ext && python setup.py install && cd ..
fireantsRegistration --help
```

## Tutorial
To check out some of the tutorials, check out the `tutorials/` directory for usage.
Alternatively, to reproduce the results in the [paper](https://arxiv.org/abs/2404.01249) checkout the `fireants/scripts/` directory.

## Fused CUDA Operations
If you want to use the fast and memory efficient fused CUDA operations, you can install the `fireants_fused_ops` package. See [fused_ops/README.md](fused_ops/README.md) for a basic user guide.

## CLI Tools
FireANTs provides command-line interface tools similar to the original ANTs toolkit. For detailed instructions and available tools, see [cli/README.md](cli/README.md).

## Template building
FireANTs includes a powerful template builder for creating anatomical templates from medical images. For detailed instructions, configuration options, and usage examples, see [fireants/scripts/template/README.md](fireants/scripts/template/README.md).

## Documentation
You can also check out the [Documentation](https://fireants.readthedocs.io/en/latest/).

## Datasets
In the paper, we use the datasets as following:
* Klein's evaluation of 14 non-linear registration algorithms: [here](https://www.synapse.org/#!Synapse:syn3251018)
* EMPIRE10 lung registration challenge: [here](https://empire10.grand-challenge.org/)
* Expansion Microscopy dataset: [here](https://rnr-exm.grand-challenge.org/)

## Tests

```
python run_tests.py
```

## License

See [LICENSE](LICENSE). Derivative work of
[FireANTs](https://github.com/rohitrango/fireants).
