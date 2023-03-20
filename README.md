# flaregun

A small PyTorch helper package to get stats on GPU usage, model params, etc.

## Installation

```bash
$ pip install flaregun
```

## Usage

Get real-time Nvidia GPU memory usage in Python:

```python
from flaregun import GPUStats

# Choose a GPU to measure
device = 0

# Get free GPU memory (in MB)
free_mem = GPUStats(device).free()
# Get total GPU memory (in MB)
total_mem = GPUStats(device).total()
# Get used GPU memory (in MB)
used_mem = GPUStats(device).used()

assert used_mem + free_mem == total_mem, "Free + Used = Total"

# Pretty print GPU statistics
GPUStats(0).print()
> "GPU memory usage: 3061 / 32510 MB"
```

Get parameter count in any PyTorch compatible model (e.g. HuggingFace, etc.):

```python
from flaregun import ModelStats

model = AutoModelForMaskedLM.from_pretrained(path_to_model)

# Get number of total params
total = ModelStats(model).total()
# Get number of trainable params
trainable = ModelStats(model).trainable()
# Get number of frozen (non-trainable) params
frozen = ModelStats(model).frozen()

assert trainable + frozen == total, "Trainable + Non-Trainable = Total"


# Pretty print Model parameter count
ModelStats(model).print()
> "148711257 params (148711257 trainable | 0 non-trainable)"
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

### Running Tests

```bash
poetry run pytest tests/
```

### Building Docs

```bash
cd docs
poetry run make clean html && poetry run make html
```
## License

`flaregun` was created by Michael Wornow. It is licensed under the terms of the MIT license.

## Credits

`flaregun` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
