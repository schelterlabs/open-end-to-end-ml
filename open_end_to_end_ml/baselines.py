from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from open_end_to_end_ml.steps import DataAugmentation, DataCleaning, FeatureAdder, ModelTrainer


class NoDataAugmentation(DataAugmentation):
    def augment(self, data):
        return data


class NoDataCleaning(DataCleaning):
    def clean(self, data):
        return data


class ArticlesToInvestigateTextualFeatureAdder(FeatureAdder):
    def add(self, data):
        data['title_and_description'] = data[['title', 'description']].apply(lambda x: ' '.join(x), axis=1)
        return data


class ArticlesToInvestigateLogisticRegressionTrainer(ModelTrainer):
    def create_pipeline(self):
        categorical_attributes = ['ActionGeo_Fullname', 'Actor1Name', 'Actor2Name', 'site_name']

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), 'title_and_description')
        ], sparse_threshold=1.0)

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        return GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)