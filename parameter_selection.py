from sklearn.model_selection import GridSearchCV


class ParameterSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" %
                missing_params)

        self.models = models
        self.params = params
        self.keys = models.keys()

    def fit(self, X, y, key, cv=5):
        print("Running GridSearchCV for %s." % key)
        model = self.models[key]
        params = self.params[key]
        gridsearch = GridSearchCV(model, params, cv=cv)
        gridsearch.fit(X, y)
        return gridsearch
