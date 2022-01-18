from datetime import datetime


def date_parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')