from xgboost import XGBClassifier
from util.train import precision, recall

class XGBoost:

    """
    :param max_depth:树的深度，默认为6.值过大易过拟合，值过小易欠拟合
    :param n_estimators:基学习器的个数，默认值是100
    :param  random_state:随机种子
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, random_state: int = 1):
        self.xgboost = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, use_label_encoder=False)

    #训练
    def train(self, x, y):
        self.xgboost.fit(x, y)

    #测试
    def test(self, x, y):
        #y_pred:当前x下y的预测值
        y_pred = self.xgboost.predict(x)
        #length为集合总数
        length = len(y)
        #采用二值分类的混淆矩阵作为评价标准
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

        #打印验证集上:正确率val_acc 查准率val_precision 查全率(召回率)val_recall
        print('val_acc: %f  val_precision: %f  val_recall: %f' % ((tp + tn) / length, precision(tp, fp), recall(tp, fn)))

    def predict(self, x):
        return self.xgboost.predict(x)