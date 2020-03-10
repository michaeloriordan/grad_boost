import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from GradientBoosting import GradientBoostingClassifier


def test():
    data = pd.read_csv('data/creditcard.csv')
    X = data.drop('Class', axis=1).values
    y = data['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42,
                                                        stratify=y)

    gbc = GradientBoostingClassifier(max_split_values=10).fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    print(classification_report(y_test, y_pred>=0.5))


if __name__ == '__main__':
    test()
