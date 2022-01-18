from datetime import datetime


def date_parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__