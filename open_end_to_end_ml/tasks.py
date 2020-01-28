from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


class ArticlesToInvestigateTask(ABC):

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

    @abstractmethod
    def augment(self, prepared_data):
        pass

    @abstractmethod
    def clean(self, augmented_data):
        pass

    @abstractmethod
    def create_pipeline(self):
        pass

    def run(self):
        name = self.__class__.__name__
        past_days = []
        current_day = 20191001

        print("name\tday\ttrain_score\ttest_score")

        for _ in range(0, 30):
            past_days.append(current_day)
            current_day += 1

            train_score, test_score = self.__run_single(past_days, current_day)
            print(f"{name}\t{current_day}\t{train_score}\t{test_score}")

    def __run_single(self, past_days, next_day):
        raw_train_data = self.__load_raw_data(past_days)
        raw_test_data = self.__load_raw_data([next_day])

        train_data = self.__integrate(raw_train_data)
        test_data = self.__integrate(raw_test_data)

        augmented_train_data = self.augment(train_data)
        augmented_test_data = self.augment(test_data)

        cleaned_train_data = self.clean(augmented_train_data)
        cleaned_test_data = self.clean(augmented_test_data)

        train_labels = self.__extract_labels(cleaned_train_data)
        test_labels = self.__extract_labels(cleaned_test_data)

        pipeline = self.create_pipeline()

        model = pipeline.fit(cleaned_train_data, train_labels)

        train_score = self.__score(model, cleaned_train_data, train_labels)
        test_score = self.__score(model, cleaned_test_data, test_labels)

        return train_score, test_score
