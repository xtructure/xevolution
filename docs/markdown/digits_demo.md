# Evolving a small neural network for Digits

This example uses xevolution to choose the architecture of a small multilayer
perceptron (MLP). Training learns each network's weights. Evolution changes its
hidden layers and activation, trains the resulting candidate, and compares how
many validation images it classifies correctly.

The goal is an understandable example of the framework. **Fitness is accuracy
alone:** the number of correct validation predictions divided by the number of
validation images. A fitness of `0.925` means 92.5% correct. There is no loss,
parameter-count penalty, or size preference hidden in this score.

The [package overview](overview.md) explains the shared evolution cycle. For
simpler representations and the other evaluation hook, see the
[arithmetic](arithmetic_demo.md) and [shared-resource](batch_demo.md) guides,
or start with the [examples comparison](examples.md).

## 1. Install and run

Run these commands from the source project's root, beside `pyproject.toml`.
Use Python 3.12 as the initial baseline. Every dependency is installed in the
repository-local virtual environment created and managed by uv.

```bash
uv python install 3.12
uv sync --locked --group dev --group neural
uv run --locked --group neural python -m xevolution run --config examples/neural_digits_quick.json --output runs/digits-quick
uv run --locked --group neural python -m examples.digits_report runs/digits-quick
```

Keep `UV_PROJECT_ENVIRONMENT` unset or pointing to this repository's `.venv`.
Keep `--group neural` on commands that build, train, or restore this application.
The `neural` group declares NumPy, scikit-learn, and threadpoolctl; uv records
their resolved transitive dependencies in `uv.lock`. They are not dependencies
of the installed xevolution core wheel.

The Digits data ships with scikit-learn, so running the example needs no further
dataset download. Once the uv environment is prepared, training needs neither
a network connection nor a GPU.

The run command prints a JSON summary containing `best_fitness`. The companion
report displays the selected architecture and its accuracy as both counts and
percentages, alongside a fixed one-hidden-layer, 32-neuron ReLU baseline.

For the normal preset, use a fresh output directory:

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits.json --output runs/digits
uv run --locked --group neural python -m examples.digits_report runs/digits --output reports/digits.json
```

`--output` on the report command is an optional JSON file outside the run
directory. The command still prints its readable summary. The report validates
the saved run and requires a completed evolutionary run before evaluating its
test images.

## 2. What evolves?

An individual carries an immutable architecture record. Its explicit JSON codec
uses this representation:

```json
{
  "hidden_layers": [32, 16],
  "activation": "relu"
}
```

Each number is the number of individual neurons in that hidden layer. This
example means 64 inputs, a hidden layer with 32 neurons, a hidden layer with
16 neurons, and ten output class scores. Every neuron in a layer receives all
outputs from the preceding layer. A unit means one neuron, not a block of
neurons.

The allowed search space is intentionally small:

| Property | Allowed values |
| --- | --- |
| Number of hidden layers | 1, 2, or 3 |
| Neurons in each hidden layer | 8, 16, 32, or 64 |
| Hidden activation | `relu` or `tanh`, shared across hidden layers |
| Input size | Fixed at 64 pixel values |
| Output size | Fixed at ten digit classes |

The largest architecture, `64 -> 64 -> 64 -> 64 -> 10`, has 13,130 trainable
weights and biases. The baseline, `64 -> 32 -> 10`, has 2,410.

Mutation resizes a layer, inserts a layer, removes a layer, or switches the
activation. Inapplicable operations, such as removing the only hidden layer,
use the framework's defined operator-inapplicability mechanism. Crossover
combines parent descriptions within the same architecture bounds. Parents are
never modified, and the framework assigns fresh child IDs and genealogy.

The normal and quick presets use tournament parent selection with a tournament
size of three and retain one elite individual. They select mutation with
probability 0.7 and crossover with probability 0.3. Fitness ties follow the
framework's stable policies; model size does not break ties.

Retaining an elite preserves its architecture and identity. Its network is
still trained afresh when the next population is evaluated, as required by
xevolution's population-evaluation contract.

## 3. What happens during one evaluation?

The application prepares a fixed stratified division of the 1,797 examples:

| Partition | Examples | Purpose |
| --- | ---: | --- |
| Training | 1,000 | Learn weights using the selected architecture |
| Validation | 400 | Compute the fitness used by evolution |
| Final test | 397 | Evaluate the selected model after evolution finishes |

Pixel values are divided by their known maximum, 16. This transformation has
no fitted statistics and uses no validation/test information. The split indices
and data contents are fingerprinted for checkpoint compatibility.

For each candidate, the evaluator:

1. Creates a fresh scikit-learn `MLPClassifier` from the architecture.
2. Initializes its weights with the recorded `training_seed`.
3. Trains for exactly the configured number of epochs with Adam, fixed learning
   rate and regularization, and the same precomputed minibatch order.
4. Predicts the validation labels and counts correct predictions.
5. Returns accuracy as fitness, with numeric measurements describing the score
   and architecture.

The initialization generator and the generator that produces epoch
permutations are separate. Different architectures consume different numbers
of random weights, so a shared initialization seed alone would not ensure the
same ordering of training samples. Precomputed ordering keeps that part of the
comparison fixed. Evolution uses its own managed random stream.

The training loop makes one `partial_fit` call per epoch, preserving the
candidate's Adam state between epochs. It does not reuse a trained network
between candidate evaluations. The batch size is capped at the training
partition's size for small test configurations. Numerical work uses one BLAS
thread, and there is no candidate-level parallel executor or fitness cache.

Training loss is used internally by the classifier to learn weights, as usual;
it is not returned as fitness. Reported size and depth are descriptive metrics.
Elapsed time is operational information and does not enter scientific history
or evolutionary decisions.

## 4. Presets and configurable values

| Preset | Population | Generations evaluated | Epochs per candidate | Candidate evaluations |
| --- | ---: | --- | ---: | ---: |
| Quick | 6 | 0 through 3 | 10 | 24 |
| Normal | 8 | 0 through 5 | 30 | 48 |

Generation 0 is the initial evaluated population. With a fixed population size
`P` and final generation `G`, evaluation count is `P * (G + 1)`, including elites.
The final report performs two additional model fits outside this count.

Application parameters live under `experiment.parameters` in the JSON file:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `train_size` | 1000 | Number of training images |
| `validation_size` | 400 | Number of images used to calculate fitness |
| `test_size` | 397 | Number of final held-out test images |
| `epochs` | 30 | Complete passes through training data per candidate |
| `batch_size` | 128 | Maximum examples in one training minibatch |
| `learning_rate` | 0.001 | Fixed initial Adam learning rate |
| `alpha` | 0.0001 | scikit-learn's L2 regularization coefficient |
| `split_seed` | 17 | Seed used to create the fixed stratified split |
| `training_seed` | 23 | Common model-initialization and derived shuffle seed |

Split sizes must add to 1,797 and must support all ten classes. Counts and seeds
must be integers, excluding booleans. Learning rate and regularization must be
finite and within their documented validation ranges; unknown parameter names
are errors. Read `DigitsSettings` in the example for exact programmatic bounds.

The top-level `seed` controls the evolutionary run. Changing it explores a
different evolutionary trajectory while the explicit split/training seeds
continue to define the same scoring task. Change a configuration only when
starting a new run; resume uses the saved configuration.

## 5. Read the results

```bash
uv run --locked --group neural python -m xevolution inspect runs/digits
uv run --locked --group neural python -m xevolution analyze runs/digits --output reports/digits-history
uv run --locked --group neural python -m examples.digits_report runs/digits --output reports/digits.json
```

`inspect` reads checkpoint metadata. `analyze` exports the existing scientific
history without training or evaluating anything. `digits_report` is a separate
application-level final evaluation: it trains the recorded historical-best
architecture afresh and verifies that its validation accuracy reproduces the
saved score. It also trains the fixed 32-neuron ReLU baseline with the same
training split and budget.

Both architectures are fixed before final test prediction. The report computes
each model's test accuracy once, with no selection based on those test scores.
It does not train on the combined training-plus-validation set. This makes the
reported comparison match the training protocol that evolution optimized.

The report never advances the run, changes the committed journal, or appends
its new evaluations to scientific history. Repeating the report in a compatible
environment reproduces its results; it does retrain the two models. Optional
report output must be outside the run directory.

A run need not beat the fixed baseline to demonstrate the framework correctly.
With only 400 validation images, one changed prediction moves accuracy by
0.25 percentage points. Short training budgets also favor architectures that
learn well within that budget; they do not establish which architecture would
win after convergence. Treat small differences as an illustration, not evidence
of a generally superior classifier.

## 6. Pause and resume

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits.json --output runs/digits-paused --pause-after-generation 2
uv run --locked --group neural python -m xevolution resume runs/digits-paused
```

The engine checkpoints every completed generation, including generation 0.
Restore reconstructs and validates the data, split, recipe, and numerical
environment; restores explicit evaluator state; and resumes the saved
evolutionary random streams. It does not train the saved generation again.

The evaluator counts completed model fits through the explicit component-state
contract. Prepared arrays are reconstructed from the bundled dataset and
recorded settings, with integrity fingerprints. Trained estimators and their
optimizer states are temporary evaluation work and are not placed in JSON
checkpoints.

If a crash interrupts a candidate fit or an uncommitted generation, that work
can run again from the saved checkpoint. Fixed seeds and compatible numerical
execution reproduce the same accepted scores. The guarantee requires matching
application code, data, Python/library versions, and relevant numerical backend
settings. It does not promise identical trajectories across different library
versions, operating systems, or processor numerical implementations.

The first `Ctrl-C` requests a pause after the current generation completes.
Ordinary resume honors a saved terminal decision. Changing a generation limit
in the original JSON file does not extend an already completed run.

## 7. Runtime and verification

The architecture and data bounds are intended to keep CPU evaluation short.
Actual measurements for this release are in [verification.md](verification.md).
They are measurements of the stated execution environment, not a guarantee for
every laptop. Time the full evaluation, including model construction and
validation, when calibrating a different machine or training recipe.

There is no wall-clock timeout that assigns a different fitness to a slow
candidate. Set a small, fixed epoch budget before a run and leave margin below
your time target. At five seconds per candidate the normal preset would take
about four minutes for its 48 evaluations; at 30 seconds it would take about
24 minutes, before the final report.

Run the optional behavioral and continuation tests with:

```bash
uv run --locked --group dev --group neural python -m pytest -m neural
```

The tests exercise real small MLP fits, accuracy accounting, immutable
architectures, variation bounds, strict settings, checkpoint compatibility,
fresh-process continuation, and final reporting. The core test suite can still
run with the development group alone.

## Source references

- [scikit-learn Digits loader and dataset description](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
- [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), the source of scikit-learn's bundled subset.
- [MLPClassifier parameters and incremental training](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
- [scikit-learn parallelism and numerical thread control](https://scikit-learn.org/stable/computing/parallelism.html).
- [uv dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups).
