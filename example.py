from open_end_to_end_ml.tasks import ArticlesToInvestigateTask
from open_end_to_end_ml.baselines import NoDataAugmentation, NoDataCleaning, \
    ArticlesToInvestigateTextualFeatureAdder, ArticlesToInvestigateLogisticRegressionTrainer


experiment = ArticlesToInvestigateTask(
    random_state=42,
    run_name='baseline',
    augmentation=NoDataAugmentation(),
    cleaning=NoDataCleaning(),
    feature_adder=ArticlesToInvestigateTextualFeatureAdder(),
    model_trainer=ArticlesToInvestigateLogisticRegressionTrainer()
)

experiment.run()

