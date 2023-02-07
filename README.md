# InfluentialFairness
The code is released exclusively for review purposes with the following terms: PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED USE, COPYING, OR DISTRIBUTION OF THE CODE, VIA ANY MEDIUM, IS STRICTLY PROHIBITED. BY ACCESSING THE CODE, THE REVIEWERS AGREE TO DELETE THEM FROM ALL MEDIA AFTER THE REVIEW PERIOD IS OVER.

Folder containing pipeline scripts for bias mitigation via Influential Fairness.

The code is set up for COMPAS, LAW, or ADULT dataset, but can easily generalize to other tabular, text, and image datasets.

influence_functions.py: contains all the functions needed for producing and computing scores

InfluentialFairness.py: script that computes influence scores on example datasets. First script to be run on formatted dataset. From here, any of the next set of scripts can be run.

singleStep_mitigationFxn.py and iterative_mitigationFxn.py: scripts to conduct mitigation using proposed strategies

InfluentialFairness, singleStep and iterative functions must be run with config file that includes:

[SETTINGS]

data_name = DATA_NAME

outcome_column_name = COLUMN_NAME

group_column_name = COLUMN_NAME

default_minority_group = ATTRIBUTE_VALUE

default_majority_group = ATTRIBUTE_VALUE
