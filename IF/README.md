# InfluentialFairness
The code is released exclusively for review purposes with the following terms: PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED USE, COPYING, OR DISTRIBUTION OF THE CODE, VIA ANY MEDIUM, IS STRICTLY PROHIBITED. BY ACCESSING THE CODE, THE REVIEWERS AGREE TO DELETE THEM FROM ALL MEDIA AFTER THE REVIEW PERIOD IS OVER.

Folder containing pipeline scripts for bias mitigation via Influential Fairness.

The code is set up for COMPAS, LAW, or ADULT dataset, but can easily generalize to other tabular, text, and image datasets.

pytorch_functions.py: contain functions related to creating dataset objects and models via PyTorch

BB_functions.py: contain functions to create Black-Box Influence Scores

WB_functions.py: contain functions to create Ground truth Influence Scores

ruleClassification_functions.py: contain functions to create rule-based explanations of given scores

mitigationMethods.py: contain main proposed mitigation methods

RBO_functions.py: Contain RBO metric

support_functions.py: Contain additional support functions.

Environment:
- Python version: 3.10.4
- Other Package details can be found in src/req.txt


