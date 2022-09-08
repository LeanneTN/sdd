#查准率 精确率
def precision(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_positive)

#查全率
def recall(true_positive, false_negative):
    if true_positive + false_negative == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_negative)

#查全率和查准率的调和平均数
def f_measure(precision, recall):
    if precision + recall == 0:
        return 0.0
    else:
        return (2 * precision * recall) / (precision + recall)