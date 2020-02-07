from open_end_to_end_ml.tasks import ArticlesToInvestigateTask
from open_end_to_end_ml.baselines import NoDataAugmentation, \
    ArticlesToInvestigateTextualFeatureAdder, ArticlesToInvestigateLogisticRegressionTrainer

from open_end_to_end_ml.steps import DataCleaning


class ModeImputation(DataCleaning):

    def __init__(self, columns):
        self.columns = columns
        self.most_frequent_values_per_column = {}

    def __find_most_frequent_values(self, data):
        for column in self.columns:
            column_without_empty_strings = data[data[column] != ''][column]
            self.most_frequent_values_per_column[column] = column_without_empty_strings.mode()[0]

    def clean(self, data, random_state, is_train):
        if is_train:
            self.__find_most_frequent_values(data)

        imputed_data = data.copy(deep=True)

        for column in self.columns:
            value_to_impute = self.most_frequent_values_per_column[column]
            imputed_data[column].replace('', value_to_impute, inplace=True)

        return imputed_data


imputer = ModeImputation(['ActionGeo_Fullname', 'Actor1Name', 'Actor2Name', 'site_name'])


experiment = ArticlesToInvestigateTask(
    random_state=42,
    run_name='mode_imputation',
    augmentation=NoDataAugmentation(),
    cleaning=imputer,
    feature_adder=ArticlesToInvestigateTextualFeatureAdder(),
    model_trainer=ArticlesToInvestigateLogisticRegressionTrainer()
)

experiment.run()

