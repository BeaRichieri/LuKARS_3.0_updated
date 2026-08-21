# LuKARS 3.0 – Morris sensitivity and GLUE workflow

This folder contains the LuKARS 3.0 Morris sensitivity-analysis and
uncertainty-analysis exercise. The code follows a separation of responsibilities:

- `Input/config.ini` defines model settings and parameter ranges;
- `lukars_functions.py` contains the reusable LuKARS equations and helper functions;
- `main.ipynb` runs the model, Morris analysis, performance evaluation and GLUE-style uncertainty analysis;
- `Output/` stores generated results.

## Project structure

```text
LuKARS_Morris_GLUE_Binder/
├── main.ipynb
├── lukars_functions.py
├── requirements.txt
├── README.md
├── Input/
│   ├── config.ini
│   └── input_I_ET_SM_2001_2020.txt
└── Output/
    └── .gitkeep
```

## Analysis workflow

The notebook:

1. loads the input time series;
2. reads deterministic settings and stochastic parameter ranges from `Input/config.ini`;
3. constructs the Morris parameter space;
4. generates the Morris sample;
5. runs the LuKARS model for every realization;
6. calculates the Nash–Sutcliffe efficiency (NSE);
7. performs the Morris sensitivity analysis;
8. plots the Morris `mu*` and `sigma` results;
9. identifies and plots the realization with the highest NSE;
10. selects realizations with NSE > 0.5;
11. calculates and plots uncertainty bands from the selected realizations.

For Binder, the ensemble loop retains only `Q_sim`, because this is the model
output needed by the subsequent NSE, Morris and uncertainty analyses. This
reduces memory use without changing the model calculations or the analyses.

## Configuration

All model settings and parameter ranges remain in:

```text
Input/config.ini
```

The configuration includes deterministic parameters, stochastic bounds for
`kMS`, hydrotope-specific parameter bounds, and the number of Morris
trajectories and levels.

## Dependencies

This exercise requires:

- NumPy
- pandas
- Matplotlib
- Numba
- seaborn
- SALib

When this exercise is stored in the same GitHub repository as the simpler
Baget exercise, Binder uses the **single shared environment located at the
repository root**:

```text
binder/
├── requirements.txt
└── runtime.txt
```

Do not create a second `binder/` directory inside this exercise folder.

## Run online with Binder

Use the following values at **mybinder.org**:

```text
Repository provider: GitHub
GitHub repository:   BeaRichieri/LuKARS_3.0_updated
Git ref:             HEAD
File to open:        LuKARS_Morris_GLUE_Binder/main.ipynb
```

Direct Binder launch link:

https://mybinder.org/v2/gh/BeaRichieri/LuKARS_3.0_updated/HEAD?urlpath=%2Fdoc%2Ftree%2FLuKARS_Morris_GLUE_Binder%2Fmain.ipynb

Students only need a web browser; no local Python or Anaconda installation is required.

### Important Binder note

Binder sessions are temporary. Files generated in `Output/` and changes made to
the notebook disappear when the session ends unless the student downloads them.

The first model execution also includes Numba compilation, so the first run can
take longer than subsequent model evaluations.

## Run locally

From inside the `LuKARS_Morris_GLUE_Binder` folder:

```bash
pip install -r requirements.txt
jupyter lab
```

Then open `main.ipynb` and run the cells from top to bottom.

## Outputs

The notebook writes the existing analysis products to `Output/`, including:

- sampled physical parameter values;
- simulated spring discharge (`Q_sim`);
- NSE values;
- Morris sensitivity indices;
- Morris `mu*` bar plot;
- Morris `mu*`–`sigma` scatter plot;
- best Morris realization and corresponding parameters;
- observed/simulated discharge for the best realization;
- realizations with NSE > 0.5;
- uncertainty-band tables and plots.
