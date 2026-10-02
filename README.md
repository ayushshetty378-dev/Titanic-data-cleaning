# Mini Project 1 - Titanic Survival Prediction: Data Cleaning

## Run
1. Download `train.csv` from https://www.kaggle.com/c/titanic/data and place it in the `titanic_project/` folder, right next to `titanic.py`:
   ```
   titanic_project/
   ├── titanic.py
   ├── README.md
   └── train.csv        <-- put it here
   ```
   (If you keep it elsewhere, pass the path instead: `python titanic.py path/to/train.csv`)
2. `pip install pandas scikit-learn matplotlib seaborn`
3. `python titanic.py`

Outputs: `titanic_cleaned.csv`, `figures/age_distribution.png`, `figures/age_by_survival.png`

## Cleaning decisions
| Column | Issue | Fix |
|---|---|---|
| Cabin | ~77% missing | dropped |
| Age | ~20% missing, skewed | median imputation |
| Embarked | 2 missing | mode imputation |
| Sex | text | LabelEncoder (female=0, male=1) |
| Embarked | nominal text | OneHotEncoder -> Embarked_C/Q/S |
| PassengerId, Name, Ticket | no predictive value | dropped |

> Note: if `train.csv` is absent the script falls back to synthetic data with the same schema (clearly warned in output) so the pipeline can be tested.


