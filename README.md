# Mini Project 1 - Titanic Survival Prediction: Data Cleaning

## Run
1.Download train.csv from https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv and save it as train.csv next to titanic.py
   ```
   titanic_project/
   ├── titanic.py
   ├── README.md
   └── train.csv        <-- put it here
   ```
   
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




