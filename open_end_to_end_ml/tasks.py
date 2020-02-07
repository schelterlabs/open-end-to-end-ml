import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score


class ArticlesToInvestigateTask:

    def __init__(self, random_state, run_name, augmentation, cleaning, feature_adder, model_trainer):
        self.random_state = random_state
        self.run_name = run_name
        self.augmentation = augmentation
        self.cleaning = cleaning
        self.feature_adder = feature_adder
        self.model_trainer = model_trainer

    def __load_raw_data(self, days):
        return {
            'events': self.__load_events(days),
            'articles': self.__load_articles(days)
        }

    def __load_events(self, days):
        daily_events = [pd.read_csv(f'data/articles-to-investigate/events/{day}.csv', index_col=0) for day in days]
        return pd.concat(daily_events, ignore_index=True)

    def __load_articles(self, days):
        daily_articles = [pd.read_csv(f'data/articles-to-investigate/articles/{day}.csv', index_col=0) for day in days]
        return pd.concat(daily_articles, ignore_index=True)

    def __integrate(self, data):
        joined_data = data['events'].merge(data['articles'], left_on='SOURCEURL',
                                           right_on='source_url', how='left')
        joined_data = joined_data.fillna('')
        return joined_data

    def __extract_labels(self, prepared_data):
        return np.ravel(label_binarize(prepared_data['investigate'], [True, False]))

    def __score(self, model, data, true_labels):
        predictions = model.predict_proba(data)
        return roc_auc_score(true_labels, np.transpose(predictions)[1])

    def run(self):

        np.random.seed(self.random_state)

        name = self.__class__.__name__
        past_days = []
        current_day = 20191001

        logfile_name = f'{name.lower()}-{self.run_name.lower()}-{self.random_state}-log.csv'

        with open(logfile_name, 'w') as log_file:

            log_file.write("task\trun\tseed\tday\ttrain_score\ttest_score\n")
            print("task\trun\tseed\tday\ttrain_score\ttest_score")

            for _ in range(0, 30):
                past_days.append(current_day)
                current_day += 1

                train_score, test_score = self.__run_single(past_days, current_day)
                log_file.write(f"{name}\t{self.run_name}\t{self.random_state}\t{current_day}\t{train_score}\t{test_score}\n")
                print(f"{name}\t{self.run_name}\t{self.random_state}\t{current_day}\t{train_score}\t{test_score}")


    def __run_single(self, past_days, next_day):
        raw_train_data = self.__load_raw_data(past_days)
        raw_test_data = self.__load_raw_data([next_day])

        train_data = self.__integrate(raw_train_data)
        test_data = self.__integrate(raw_test_data)

        augmented_train_data = self.augmentation.augment(train_data, self.random_state, is_train=True)
        augmented_test_data = self.augmentation.augment(test_data, self.random_state, is_train=False)

        cleaned_train_data = self.cleaning.clean(augmented_train_data, self.random_state, is_train=True)
        cleaned_test_data = self.cleaning.clean(augmented_test_data, self.random_state, is_train=False)

        final_train_data = self.feature_adder.add(cleaned_train_data, self.random_state, is_train=True)
        final_test_data = self.feature_adder.add(cleaned_test_data, self.random_state, is_train=False)

        train_labels = self.__extract_labels(final_train_data)
        test_labels = self.__extract_labels(final_test_data)

        pipeline = self.model_trainer.create_pipeline(self.random_state)

        model = pipeline.fit(final_train_data, train_labels)

        train_score = self.__score(model, final_train_data, train_labels)
        test_score = self.__score(model, final_test_data, test_labels)

        return train_score, test_score
