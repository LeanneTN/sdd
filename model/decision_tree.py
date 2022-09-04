from sklearn.tree import DecisionTreeClassifier
from util.train import precision, recall


class DecisionTree:

    def __init__(self):
        self.dt = DecisionTreeClassifier()

    def train(self, x, y):
        self.dt.fit(x, y)

    def test(self, x, y):
        y_pred = self.dt.predict(x)
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

        # 打印验证集上:正确率val_acc 查准率val_precision 查全率(召回率)val_recall
        print(
            'val_acc: %f  val_precision: %f  val_recall: %f' % ((tp + tn) / length, precision(tp, fp), recall(tp, fn)))

    def predict(self, x):
        return self.dt.predict(x)