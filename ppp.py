import pandas as pd
import numpy as np
import random
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

class SwappedValues:

    def __init__(self, fraction, column_pair):
        self.fraction = fraction
        self.column_pair = column_pair

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        (column_a, column_b) = self.column_pair

        values_of_column_a = list(df[column_a])
        values_of_column_b = list(df[column_b])

        for index in range(0, len(values_of_column_a)):
            if random.random() < self.fraction:
                temp_value = values_of_column_a[index]
                values_of_column_a[index] = values_of_column_b[index]
                values_of_column_b[index] = temp_value

        df[column_a] = values_of_column_a
        df[column_b] = values_of_column_b

        return df

class Outliers:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)
        # means = {column: np.mean(df[column]) for column in self.columns}
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: random.uniform(1, 5) for column in self.columns}

        if self.fraction > 0:
            for column in self.columns:
                rows = np.random.uniform(size=len(df))<self.fraction
                noise = np.random.normal(0, scales[column] * stddevs[column], size=rows.sum())
                df.loc[rows, column] += noise

        return df


class Scaling:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])
        
        if self.fraction > 0:
            for column in self.columns:
                rows = np.random.uniform(size=len(df))<self.fraction
                df.loc[rows, column] *= scale_factor

        return df

class MissingValuesHighEntropy:

    def __init__(self, 
                    fraction, 
                    model, 
                    categorical_columns, 
                    numerical_columns,
                    categorical_value_to_put_in='NULL',
                    numerical_value_to_put_in=0):
        self.fraction = fraction
        self.model = model
        self.categorical_value_to_put_in = categorical_value_to_put_in
        self.numerical_value_to_put_in = numerical_value_to_put_in
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        probas = self.model.predict_proba(df)
        # for samples with the smallest maximum probability 
        # the model is most uncertain
        cutoff = int(len(df) * (1-self.fraction))
        least_confident = probas.max(axis=1).argsort()[-cutoff:]
        df.loc[df.index[least_confident], self.categorical_columns] = self.categorical_value_to_put_in
        df.loc[df.index[least_confident], self.numerical_columns] = self.numerical_value_to_put_in

        return df

class MissingValuesLowEntropy:

    def __init__(self, 
                    fraction, 
                    model, 
                    categorical_columns, 
                    numerical_columns,
                    categorical_value_to_put_in='NULL',
                    numerical_value_to_put_in=0):
        self.fraction = fraction
        self.model = model
        self.categorical_value_to_put_in = categorical_value_to_put_in
        self.numerical_value_to_put_in = numerical_value_to_put_in
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        probas = self.model.predict_proba(df)
        # for samples with the smallest maximum probability 
        # the model is most uncertain
        cutoff = int(len(df) * (1-self.fraction))
        most_confident = probas.max(axis=1).argsort()[:cutoff]
        for c in self.categorical_columns:
            df.loc[df.index[most_confident], c] = self.categorical_value_to_put_in
        for c in self.numerical_columns:
            df.loc[df.index[most_confident], c] = self.numerical_value_to_put_in

        return df

class PipelineWithPPP:

    def __init__(self, 
                pipeline, 
                num_repetitions=10, 
                perturbation_fractions=np.linspace(0,1,11)):
        self.pipeline = pipeline
        self.num_repetitions = num_repetitions
        self.perturbation_fractions = perturbation_fractions
        # assuming the first step is a ColumnTransformer with transformers named 
        # 'categorical_features' or 'numerical_features'
        self.categorical_features = []
        self.numerical_features = []
        for t in pipeline.steps[0][1].transformers:
            if t[0]=='categorical_features':
                self.categorical_features = t[2]
            if t[0]=='numerical_features':
                self.numerical_features = t[2]
        print(f'Registered categorical columns: {self.categorical_features}')
        print(f'Registered numerical columns: {self.numerical_features}')

        
        self.perturbations = []
        for _ in range(self.num_repetitions):
            for fraction in self.perturbation_fractions:
                
                numerical_column_pairs = list(itertools.combinations(self.numerical_features, 2))
                swap_affected_column_pair = random.choice(numerical_column_pairs)
                affected_numeric_column = random.choice(self.numerical_features)
                affected_categorical_column = np.random.choice(self.categorical_features)

                self.perturbations += [
                    ('swapped', SwappedValues(fraction, swap_affected_column_pair)),
                    ('scaling', Scaling(fraction, [affected_numeric_column])),
                    ('outlier', Outliers(fraction, [affected_numeric_column])),
                    ('missing_high_entropy', MissingValuesHighEntropy(fraction, pipeline, [affected_categorical_column], [affected_numeric_column])),
                    ('missing_low_entropy', MissingValuesLowEntropy(fraction, pipeline, [affected_categorical_column], [affected_numeric_column])),
                ]


    @staticmethod
    def compute_ppp_features(predictions):
        probs_class_a = np.transpose(predictions)[0]
        features_a = np.percentile(probs_class_a, np.arange(0, 101, 5))
        if predictions.shape[-1] > 1:
            probs_class_b = np.transpose(predictions)[1]
            features_b = np.percentile(probs_class_b, np.arange(0, 101, 5))
            return np.concatenate((features_a, features_b), axis=0)
        else:
            return features_a

    def fit_ppp(self, X_df, y):

        print("Generating perturbed training data...")
        meta_features = []
        meta_scores = []
        for perturbation in self.perturbations:
            df_perturbed = perturbation[1](X_df)
      
            predictions = self.pipeline.predict_proba(df_perturbed)
            
            meta_features.append(self.compute_ppp_features(predictions))
            meta_scores.append(self.pipeline.score(df_perturbed, y))
 
        param_grid = {
            'learner__n_estimators': np.arange(5, 20, 5),
            'learner__criterion': ['mae']
        }

        meta_regressor_pipeline = Pipeline([
           ('scaling', StandardScaler()),
           ('learner', RandomForestRegressor(criterion='mae'))
        ])

        print("Training performance predictor...")
        self.meta_regressor = GridSearchCV(
                                meta_regressor_pipeline, 
                                param_grid, 
                                scoring='neg_mean_absolute_error')\
                                    .fit(meta_features, meta_scores)

    def predict_ppp(self, X_df):
        meta_features = self.compute_ppp_features(self.pipeline.predict_proba(X_df))
        return self.meta_regressor.predict(meta_features.reshape(1, -1))[0]