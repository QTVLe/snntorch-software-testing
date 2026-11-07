# LO3 tests and experiments

Run these commands from the repository root in a Python 3.12 virtual environment.

```sh
python -m pip install -e . pytest coverage scipy tqdm h5py requests torchvision six
python portfolio-testing/lo3/run.py final
python portfolio-testing/lo3/mutations.py
python portfolio-testing/lo3/assess.py
```

The run script executes the existing repository tests, the LO3 suite with coverage
and a smoke test. It records each command and its exit code under
`portfolio-testing/lo3/evidence/final`. The mutation experiment checks whether
the tests detect three deliberate faults introduced in separate processes.
`assess.py` summarises the recorded test and coverage results.

To replay the downloader security tests against the original revision, run
`python portfolio-testing/lo3/original_security.py` in a checkout with the
baseline commit in its history. Test failures are expected in this replay.
`python portfolio-testing/lo3/generate.py` regenerates the test case manifests.

The runners preserve existing output directories. Use a fresh checkout to repeat
the same experiment without overwriting its previous output.
