# fatigued-face-space

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.2.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## Check an experiment before running it

Run the read-only preflight check from the project root:

```console
kedro check --env=us_2601_v2 --pipeline=v2
```

The command checks that the experiment configuration directory exists, that
`globals.yml` contains a `metadata_file` value, and that the effective
configuration from `conf/base` and `conf/<env>` provides every `params:` input
used by the selected pipeline. For pipelines that use `movie_metadata`, it also
checks that the catalog points to `metadata_<metadata_file>.csv` and that the
file exists. Multiple experiment environments can use the same `metadata_file`.
The `feature_extraction` and `external_csv` pipelines do not require metadata.

Required parameters fail the check when their effective configuration is missing,
is an empty string, or is written without a value (for example, `key:`). An
explicit `key: null` is accepted. Blank values report their source file and line.

The check reports all missing requirements it can identify, returns exit code
`0` on success and `1` on failure, and does not execute pipeline nodes. It does
not inspect metadata contents, source data, or Docker availability.
The former `kedro expt check` command remains available for existing scripts.

To inspect the values and Python types that a pipeline's nodes receive, run:

```console
kedro check params --env=us_2601_peakexpt_home1 --pipeline=v3
kedro check params peak --env=us_2601_peakexpt_home1 --pipeline=v3
kedro check params peak.min_distance_seconds --env=us_2601_peakexpt_home1 --pipeline=v3
```

The optional positional argument selects an exact parameter or its dotted group.
Values come from the merged `conf/base` and `conf/<env>` configuration. Types are
checked against node argument annotations; `int` is compatible with `float`,
while the actual type remains visible. Missing values, mismatches, and parameters
without checkable annotations cause a nonzero exit code. Runtime `kedro run
--params` overrides are not included.

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
