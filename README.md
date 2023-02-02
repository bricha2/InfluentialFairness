# InfluentialFairness
Repository with pipeline scripts for bias mitigation via Influential Fairness

influence_functions.py: contains alll the functions needed for producing and computing scores

InfluentialFairness.py: script that computes influence scores on example datasets

singleStep_mitigationFxn.py and iterative_mitigationFxn.py: scripts to conduct mitigation using proposed strategies

singleStep and iterative functions must be run with config file that includes:

[SETTINGS]

data_name = DATA_NAME

outcome_column_name = COLUMN_NAME

group_column_name = COLUMN_NAME

default_minority_group = ATTRIBUTE_VALUE

default_majority_group = ATTRIBUTE_VALUE
