from sklearn.svm import SVC
from util.train import precision, recall, f_measure

class SVM:

    def __init__(self):
        self.my_svm = SVC()

    def train(self, x, y):
        self.my_svm.fit(x, y)

    def test(self, x, y):
        y_pred = self.my_svm.predict(x)
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

        print(
            'val_acc: %f  val_precision: %f  val_recall: %f' % ((tp + tn) / length, precision(tp, fp), recall(tp, fn)))