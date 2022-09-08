import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss, BCELoss, KLDivLoss, Module
from torch.optim import SGD, Adam
from util.loss import FocalLoss, VAELoss
from util.index import log, to_json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_classification(model: Module, train_dataset: DataLoader, epochs: int, loss_function: str, optimizer: str, learning_rate: float = 1e-4, focal_loss_options: dict = None, learning_rate_decay: float = None, val_dataset: DataLoader = None, test_dataset: DataLoader = None, log_path: str = None) -> None:
    """
    模型训练函数

    :param model: 模型
    :param train_dataset: 数据集
    :param epochs: 训练轮数
    :param loss_function: 损失函数
    :param optimizer: 优化器
    :param learning_rate: 学习率
    :param focal_loss_options: Focal Loss配置
    :param learning_rate_decay: 学习率衰减
    :param val_dataset: 验证集
    :param test_dataset: 测试集
    :param log_path: 日志文件路径
    :return: None
    """
    loss_function = get_loss(loss_function, focal_loss_options)
    optimizer = get_optimizer(optimizer)(model.parameters(), lr=learning_rate)
    total_step = len(train_dataset)
    if log_path is not None:
        if os.path.exists(log_path):
            os.remove(log_path)

    for i in range(epochs):
        # 记录总正确数
        correct_total = 0
        # 记录总样本数
        total = 0
        # 记录总损失
        loss_total = 0
        # 记录总分类TP, FP, TN, FN
        true_positive_total = 0
        false_positive_total = 0
        true_negative_total = 0
        false_negative_total = 0
        print('epoch %d' % (i + 1))
        model.train()
        for step, (x, y) in enumerate(train_dataset):
            # 前向传播
            y_pred = model(x.to(device))
            # 损失计算
            loss = loss_function(y_pred, y.view(-1, 1).to(device))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            loss_total += loss
            true_positive, false_positive, true_negative, false_negative = compute_correct(y_pred, y, separate=True)
            correct_total += (true_positive + true_negative)
            total += (true_positive + false_positive + true_negative + false_negative)
            true_positive_total += true_positive
            false_positive_total += false_positive
            true_negative_total += true_negative
            false_negative_total += false_negative
            visualize_classification(step + 1, total_step, loss_total / (step + 1), correct_total / total, precision(true_positive_total, false_positive_total), recall(true_positive_total, false_negative_total))

        if val_dataset is not None:
            model.eval()
            # 验证集验证
            val_y_pred, val_y_true, val_loss = validate_classification(model, val_dataset, loss_function)
            val_true_positive, val_false_positive, val_true_negative, val_false_negative = compute_correct(val_y_pred, val_y_true, separate=True)
            val_correct = val_true_positive + val_true_negative
            val_total = val_true_positive + val_true_negative + val_false_positive + val_false_negative
            visualize_classification(total_step, total_step, loss_total / total_step, correct_total / total, precision(true_positive_total, false_positive_total), recall(true_positive_total, false_negative_total), val_loss=val_loss, val_acc=val_correct / val_total, val_precision=precision(val_true_positive, val_false_positive), val_recall=recall(val_true_positive, val_false_negative))

            if log_path is not None:
                to_json(log_path, val={
                    'acc': val_correct / val_total,
                    'precision': precision(val_true_positive, val_false_positive),
                    'recall': recall(val_true_positive, val_false_negative)
                })
        print()

    if test_dataset is not None:
        model.eval()
        # 测试集测试
        log('test dataset')
        test_y_pred, test_y_true, test_loss = validate_classification(model, test_dataset, loss_function)
        test_true_positive, test_false_positive, test_true_negative, test_false_negative = compute_correct(test_y_pred, test_y_true, separate=True)
        test_correct = test_true_positive + test_true_negative
        test_total = test_true_positive + test_true_negative + test_false_positive + test_false_negative
        print('test_loss: %f, test_acc: %f, test_precision: %f, test_recall: %f' % (test_loss, test_correct / test_total, precision(test_true_positive, test_false_positive), recall(test_true_positive, test_false_negative)))
        if log_path is not None:
            to_json(log_path, test={
                'acc': test_correct / test_total,
                'precision': precision(test_true_positive, test_false_positive),
                'recall': recall(test_true_positive, test_false_negative)
            })


def visualize_classification(step: int, total_step: int, train_loss: float, train_acc: float, train_precision: float, train_recall: float, val_loss: float = None, val_acc: float = None, val_precision: float = None, val_recall: float = None):
    info = visualize_step(step, total_step)
    info += 'loss: %f  ' % train_loss
    info += 'acc: %f  ' % train_acc
    info += 'precision: %f  ' % train_precision
    info += 'recall: %f  ' % train_recall
    if val_loss is not None:
        info += 'val_loss: %f  ' % val_loss
    if val_acc is not None:
        info += 'val_acc: %f  ' % val_acc
    if val_precision is not None:
        info += 'val_precision: %f  ' % val_precision
    if val_recall is not None:
        info += 'val_recall: %f  ' % val_recall
    print('\r%s' % info, end='', flush=True)

# vae训练文件
def fit_ae(model: Module, train_dataset: DataLoader, epochs: int, optimizer: str, val_dataset: list, test_dataset: list, learning_rate: float = 1e-4, early_stop: float = None, log_path: str = None):
    loss_function = VAELoss()
    # 初始化模型参数，确定学习率
    optimizer = get_optimizer(optimizer)(model.parameters(), lr=learning_rate)
    # 训练的总步数
    total_step = len(train_dataset)
    # 日志文件，暂时不用
    if log_path is not None:
        if os.path.exists(log_path):
            os.remove(log_path)

    # 循环查看每一轮训练的轮数
    for i in range(epochs):
        # 记录总损失
        loss_total = 0
        # 在控制台打印信息
        print('epoch %d' % (i + 1))
        # 训练模型
        model.train()
        # _可以浅显的理解为一个占位符
        for step, (x, _) in enumerate(train_dataset):
            # 前向传播
            # x_hat, norm_x, mean_x = model(x.to(device))
            # 均值和方差，这里面看forward返回了几个参数，就用几个参数来接
            x_hat, norm_x, mean_x, lv_x = model(x.to(device))
            # 损失计算
            loss = loss_function(x_hat, norm_x, mean_x, lv_x)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            loss_total += loss
            visualize_ae(step + 1, total_step, loss_total / (step + 1))
        # 训练
        model.eval()
        loss_negative, loss_positive, val_acc, val_precision, val_recall = validate_ae(model, val_dataset, loss_function)
        visualize_ae(total_step, total_step, loss_total / total_step, val_loss_negative=loss_negative, val_loss_positive=loss_positive, val_acc=val_acc, val_precision=val_precision, val_recall=val_recall)
        if log_path is not None:
            to_json(log_path, val={
                'acc': val_acc,
                'precision': val_precision,
                'recall': val_recall
            })
        if early_stop is not None and loss_total / total_step <= early_stop:
            log('early stop')
            break
    # 测试集测试
    model.eval()
    log('test dataset')
    loss_negative, loss_positive, test_acc, test_precision, test_recall = validate_ae(model, test_dataset, loss_function)
    print('test_loss_negative: %f, test_loss_positive: %f, test_acc: %f, test_precision: %f, test_recall: %f' % (loss_negative, loss_positive, test_acc, test_precision, test_recall))
    if log_path is not None:
        to_json(log_path, test={
            'acc': test_acc,
            'precision': test_precision,
            'recall': test_recall
        })

# 可视化训练结果
def visualize_ae(step: int, total_step: int, train_loss: float, val_loss_negative: float = None, val_loss_positive: float = None, val_acc: float = None, val_precision: float = None, val_recall: float = None):
    info = visualize_step(step, total_step)
    info += 'loss: %f  ' % train_loss
    if val_loss_negative is not None:
        info += 'val_loss_negative: %f  ' % val_loss_negative
    if val_loss_positive is not None:
        info += 'val_loss_positive: %f  ' % val_loss_positive
    if val_acc is not None:
        info += 'val_acc: %f  ' % val_acc
    if val_precision is not None:
        info += 'val_precision: %f  ' % val_precision
    if val_recall is not None:
        info += 'val_recall: %f  ' % val_recall
    print('\r%s' % info, end='', flush=True)

# 可视化每每一步
def visualize_step(step: int, total_step: int):
    rate = int(step * 25 / total_step)
    info = '%d/%d' % (step, total_step)
    info += ' ['
    for i in range(rate):
        info += '='
    for i in range(25 - rate):
        info += '-'
    info += '] '
    return info


def get_loss(loss: str, focal_loss_options: dict):
    if loss == 'mse':
        return MSELoss()
    elif loss == 'bce':
        return BCELoss()
    elif loss == 'kl':
        return KLDivLoss()
    elif loss == 'fl':
        if focal_loss_options is not None:
            log("using customized options: alpha--%f  gamma--%f  scale--%f  average--" % (focal_loss_options['alpha'], focal_loss_options['gamma'], focal_loss_options['scale']) + str(focal_loss_options['average']))
            return FocalLoss(alpha=focal_loss_options['alpha'], gamma=focal_loss_options['gamma'], average=focal_loss_options['average'], binary=True)
        else:
            log("using default focal loss options")
            return FocalLoss(alpha=0.5, gamma=2.0, average=True, binary=True)
    else:
        log("loss function '%s' cannot found" % loss)
        exit(1)


def get_optimizer(optimizer: str):
    if optimizer == 'sgd':
        return SGD
    elif optimizer == 'adam':
        return Adam
    else:
        log("optimizer '%s' cannot found" % optimizer)
        exit(1)

# 计算正确度
def compute_correct(y_pred, y_true, separate=False):
    length = len(y_pred)
    if separate is False:
        correct = 0
        for i in range(length):
            temp_y_pred = y_pred[i]
            temp_y_true = y_true[i]
            if temp_y_pred < 0.5 and temp_y_true == 0 or temp_y_pred > 0.5 and temp_y_true == 1:
                correct += 1
        return correct, length
    else:
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i in range(length):
            temp_y_pred = y_pred[i]
            temp_y_true = y_true[i]
            if temp_y_true == 0:
                if temp_y_pred < 0.5:
                    true_negative += 1
                else:
                    false_positive += 1
            else:
                if temp_y_pred > 0.5:
                    true_positive += 1
                else:
                    false_negative += 1
        return true_positive, false_positive, true_negative, false_negative


def validate_classification(model, dataset, loss_function):
    y_pred = []
    y_true = []
    total_step = len(dataset)
    loss = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(dataset):
            y_true += y
            temp = model(x.to(device))
            y_pred += temp
            loss += loss_function(temp, y.view(-1, 1).to(device))
    return y_pred, y_true, loss / total_step


# 这样写的话是既有正样本又有负样本
def validate_ae(model, dataset, loss_function):
    # 调用fit_ae的时候传入的参数是一个数组，0取的是negative，1取的是positive
    negative_step = len(dataset[0])
    loss_negative = 0
    positive_step = len(dataset[1])
    loss_positive = 0
    loss_n = []
    loss_p = []
    # 主要所用是停止autograd模块的工作，以起到加速和节省显存的作用，停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为
    with torch.no_grad():
        for _, (x, _) in enumerate(dataset[0]):
            x_hat, norm_x, mean_x, lv_x = model(x.to(device))
            # 更新loss_function
            loss_negative += loss_function(x_hat, norm_x, mean_x, lv_x)
            loss_n += loss_function(x_hat, norm_x, mean_x, lv_x, False).tolist()
        # 比较自编码器还原相似度损失，损失越小越可能是正样本
        for _, (x, _) in enumerate(dataset[1]):
            x_hat, norm_x, mean_x, lv_x = model(x.to(device))
            loss_positive += loss_function(x_hat, norm_x, mean_x, lv_x)
            loss_p += loss_function(x_hat, norm_x, mean_x, lv_x, False).tolist()
    negative_size = len(loss_n) #这个为什么不能直接从数据中走呢
    # print(negative_size)
    positive_size = len(loss_p)
    # print(positive_size)
    tp = 0
    fp = 0
    abnormal_size = int(model.abnormal_rate * (negative_size + positive_size))
    loss_n.sort(reverse=True)
    loss_p.sort(reverse=True)
    n_index = 0
    p_index = 0
    for _ in range(abnormal_size):
        if loss_p[p_index] >= loss_n[n_index]:#p是正样本的数量，n是负样本的数量
            tp += 1
            p_index += 1
        else:
            fp += 1
            n_index += 1
    fn = positive_size - tp
    tn = negative_size - fp
    return loss_negative / negative_step, loss_positive / positive_step, (tp + tn) / (positive_size + negative_size), precision(tp, fp), recall(tp, fn)


def precision(true_positive, false_positive):
    if true_positive + false_positive == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_positive)

#召回率
def recall(true_positive, false_negative):
    if true_positive + false_negative == 0:
        return 0.0
    else:
        return true_positive / (true_positive + false_negative)
