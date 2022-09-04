from sklearn.ensemble import RandomForestClassifier
from util.train import precision, recall


class RandomForest:

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 1):
        self.random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def train(self, x, y):
        self.random_forest.fit(x, y)

    def test(self, x, y):
        y_pred = self.random_forest.predict(x)
        length = len(y)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(length):
            if y[i] == 0:
                if y_pred[i] == 0:
                    tn += 1
                else:
                    fp += 1
            else:
                if y_pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
        print('val_acc: %f  val_precision: %f  val_recall: %f' % ((tp + tn) / length, precision(tp, fp), recall(tp, fn)))

    def predict(self, x):
        return self.random_forest.predict(x)