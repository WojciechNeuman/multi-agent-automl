from enum import Enum

class ModelEnum(Enum):
    """
    Enum-like class for model names.
    This can be extended with more models as needed.
    """
    RANDOMFOREST = "RandomForest"
    LOGISTICREGRESSION = "LogisticRegression"
    LINEARREGRESSION = "LinearRegression"
    GRADIENTBOOSTING = "GradientBoosting"
    SVC = "SVC"
    KNEIGHBORS = "KNeighbors"
