# 3 Steps to run the code

## Install prerequisites

Make sure you have `python 3.7` and `pipenv` installed on your system. To install `pipenv` run:

```bash
python3.7 -m pip install --user pipenv
```

## Install dependencies

When you are in the root directory of the project, run:

```bash
python3.7 -m pipenv --python 3.7
python3.7 -m pipenv sync
python3.7 -m pipenv shell
```

## Install dependencies (alternative way)

If you cannot install `pipenv` or just don't want to do that, use plain `pip`:

```bash
python3.7 -m pip install -r requirements.txt
```

## Run the code

After `pipenv` installs all the dependencies, you can run Task I and Task II, respectively:

```bash
python task_1.py
python task_2.py
```
