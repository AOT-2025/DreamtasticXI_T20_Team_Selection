# DreamtasticXI: Context-Aware T20I Playing XI Selection

## 1. About the Project
DreamtasticXI is a context-aware cricket team selection framework for building a T20I Playing XI from player-level batting, bowling, and fielding signals. The project combines statistical scoring, position-aware MLP models, and role-constrained team-composition logic to support both system-generated and manually configured XI selection.

## 2. Repository Contents
The repository contains the core application, supporting models, and analysis files used for prediction and team composition.

- Source code:
  - `Home.py` — main landing page for the Streamlit application
  - `pages/System_Generated.py` — system-generated XI selection interface
  - `pages/Manual_Selection.py` — manual XI configuration workflow
  - `src/statistical_score_calc.py` — statistical batting and bowling score calculation
  - `src/mlp_trainer.py` — MLP training implementation
  - `src/reliability_adjuster.py` — player reliability and role/position assignment logic
  - `src/composition_rule_engine_new.py` — context-aware role-count prediction engine
  - `src/context_sidebar_system_generated.py` — sidebar configuration for system-generated mode
  - `src/context_sidebar_manual_selection.py` — sidebar configuration for manual mode
  - `src/stylesheet.py` — Streamlit styling
  - `src/mlp_architecture_test.py` — MLP architecture sensitivity analysis
  - `src/winloss_experiment.py` — win/loss analysis and LightGBM-based composition experiment

- Data and preprocessing files:
  - `data/Players.csv` — player metadata, team, and role information
  - `data/TeamID.csv` — team identifier and ranking table
  - `data/Batting_Scores.csv` — batting performance and contextual batting scores
  - `data/Bowler_final_score.csv` — bowling performance and contextual bowling scores
  - `data/Fielding_Scores.csv` — fielding score estimates
  - `data/Batsman_Scores.csv` — batsman score table
  - `data/Bat_Position_Score.csv` — batting-position score table
  - `data/team_composition_updated_new.json` — historical team-composition knowledge base used by the composition model

- Models and analysis outputs:
  - `results/LightGBM_win_vs_winloss_*.csv` — experiment summaries and prediction outputs

## 4. Requirements
The project dependencies are listed in `requirements.txt` and include:

- Python packages:
  - `streamlit>=1.36,<2`
  - `numpy>=1.26,<3`
  - `pandas>=2.0,<3`
  - `scipy>=1.10,<2`
  - `scikit-learn>=1.3,<2`
  - `lightgbm>=4.0,<5`
  - `torch>=2.2,<3`

## 5. Installation
From a terminal in the repository root:

```bash
git clone <repository-url>
cd <repository-folder>

pip install -r requirements.txt
```

## 6. Dataset Information
The project uses structured player and match-context data for T20I team selection. The repository contains player records, opponent and team metadata, batting and bowling contextual scoring tables, fielding scores, and a historical team-composition knowledge base in `data/team_composition_updated_new.json`.

Relevant files include:

- `data/Players.csv` — player identity, role, team, and basic profile fields
- `data/TeamID.csv` — team names and ranking values
- `data/Batting_Scores.csv` — player batting contextual scores by opponent, venue, pitch, and batting position
- `data/Bowler_final_score.csv` — player bowling contextual scores by opponent, venue, and overs phase
- `data/Fielding_Scores.csv` — fielding score estimates for player selection
- `data/team_composition_updated_new.json` — historical team-composition records used to infer role-count priors and match-context patterns

These datasets are used directly by the selection pipeline; the repository does not explicitly state that they are publicly redistributable.

## 7. Data Preprocessing
The preprocessing pipeline is implemented in the code as follows:

- Match context filtering: unavailable players are removed before scoring and selection.
- Tournament/opponent handling: Series-mode scoring adjusts by opponent and recent/opponent-wide batting features; ICC mode uses position-wise medians.
- Pitch and venue conditioning: batting and bowling factors are selected based on `Pitch_Type` and `Ground` (`Home`/`Away`).
- Clutch-match feature usage: when `Clutch` is enabled, phase-specific powerplay/middle-over/death-over values are included.
- Role filtering: only valid batting or bowling roles are retained for scoring and selection.
- Feature normalization: score columns are normalized before inverse-variance weighting.
- Model input preparation: normalized features, player names, roles, and positions are assembled into training matrices for the MLP models.
- Team-composition knowledge-base preparation: the JSON composition history is transformed into role counts and rank-tier categories for the LightGBM-based composition engine.

## 8. How to Run the Framework
The interactive application is launched from the repository root with:

```bash
streamlit run Home.py
```

This starts the main Streamlit interface. The user can choose between:

- `System Generated` — automatic XI selection based on match context, predicted role counts, and model scores
- `Manual Selection` — user-defined team composition and role counts, with the system ranking suitable players

The application uses the selected tournament type, opponent, ground, pitch conditions, clutch setting, and unavailable-player list to generate a recommended final XI and associated scores.

## 9. Model/Algorithm Configuration
The repository implements a mixed pipeline with statistical scoring and neural prediction:

- Statistical scoring: inverse-variance feature weighting is used in `src/statistical_score_calc.py` to compute `True_Final_Score` and `True_Final_Bowl_Score`.
- Position-aware MLPs: `src/mlp_trainer.py` trains feed-forward networks with hidden architectures selected by position, including `[64]`, `[128]`, and `[128, 64]` variants.
- LightGBM composition model: `src/composition_rule_engine_new.py` trains a `MultiOutputRegressor` using `LGBMRegressor` to predict role counts from pitch, home/away, opponent, and rank-tier context.
- Player selection logic: `src/reliability_adjuster.py` evaluates player reliability, fielding contribution, and position suitability while respecting role and lineup constraints.
- Role and position constraints: the sidebar logic constrains the selection to an 11-player team using inferred batting, bowling, all-rounder, and wicket-keeper counts.
- Player availability handling: the user can exclude unavailable players in the app, and those players are filtered from the candidate pool before selection.

## 10. Reproducing the Experiments
The repository contains the scripts used for the experimental pipeline.

1. Install dependencies as described above.
2. Reproduce the LightGBM win/loss analysis:

```bash
python src/winloss_experiment.py
```

3. Run the interactive selection workflow:

```bash
streamlit run Home.py
```

The generated CSV artifacts in `results/`, such as `results/LightGBM_win_vs_winloss_summary.csv` and related files, are the repo-native outputs for these analyses.

## 11. Expected Outputs
The framework produces several outputs associated with the prediction and team-selection workflow:

- Player batting and bowling scores derived from contextual statistical features
- Position-wise candidate rankings for batting and bowling roles
- Predicted role counts for team composition under match context
- Final recommended Playing XI with role/position assignments
- Output tables and CSV files such as the `results/LightGBM_win_vs_winloss_*.csv` experiment results
- Streamlit visualizations showing top players per position and final XI summary

The repository therefore provides both interactive team-generation outputs and experiment-level result artefacts.
