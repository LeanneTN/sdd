import json
import os


def log(message):
    print('log info: %s' % message)


def dict_add(dict_, key, value, replace: bool = False, init: bool = False):
    if key not in dict_:
        dict_[key] = value
    elif init:
        pass
    elif replace:
        dict_[key] = value
    else:
        dict_[key] += value


def to_json(json_path: str, val: dict = None, test: dict = None):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            json_ = json.load(f)
    else:
        json_ = {}

    if val is not None:
        dict_add(json_, 'val', {}, init=True)
        if 'acc' in val:
            dict_add(json_['val'], 'acc', [val['acc']])
        if 'precision' in val:
            dict_add(json_['val'], 'precision', [val['precision']])
        if 'recall' in val:
            dict_add(json_['val'], 'recall', [val['recall']])

    if test is not None:
        dict_add(json_, 'test', {}, init=True)
        if 'acc' in test:
            dict_add(json_['test'], 'acc', test['acc'], replace=True)
        if 'precision' in test:
            dict_add(json_['test'], 'precision', test['precision'], replace=True)
        if 'recall' in test:
            dict_add(json_['test'], 'recall', test['recall'], replace=True)
        if 'precision' in test and 'recall' in test:
            dict_add(json_['test'], 'f1', 2 * test['precision'] * test['recall'] / (test['precision'] + test['recall']), replace=True)

    with open(json_path, 'w') as f:
        json.dump(json_, f, indent=4, ensure_ascii=False)
