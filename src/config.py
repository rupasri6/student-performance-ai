"""Central configuration. Change paths and modelling choices here, not in notebooks."""

from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS / "models"
FIGURE_DIR = ARTIFACTS / "figures"

MATH_CSV = DATA_RAW / "student-mat.csv"
POR_CSV = DATA_RAW / "student-por.csv"

TARGET_REGRESSION = "G3"
TARGET_CLASS = "support_need"

# G2 is almost a copy of the final grade and would leak the answer.
# G1 is a prior exam; we keep it out of the default early-warning model.
LEAKY_FEATURES = ["G1", "G2"]
FAIRNESS_EXCLUDED = ["sex"]  # not used as a predictor

NUMERIC_BASE = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
]

CATEGORICAL_BASE = [
    "school",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "subject",
]

ENGINEERED_NUMERIC = [
    "parent_edu_avg",
    "attendance_proxy",
    "alcohol_index",
    "study_vs_travel",
    "support_count",
    "engagement_score",
    "social_load",
]

ALL_FEATURES = NUMERIC_BASE + ENGINEERED_NUMERIC + CATEGORICAL_BASE

# Support bands on the 0-20 UCI grade scale (pass typically = 10).
HIGH_SUPPORT_MAX = 9.999
MEDIUM_SUPPORT_MAX = 13.999

TEST_SIZE = 0.20
CV_FOLDS = 3
N_ITER_SEARCH = 8

ABSENCE_CAP = 30  # used to turn absences into an attendance-style proxy
