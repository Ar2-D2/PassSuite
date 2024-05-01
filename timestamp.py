import datetime

def timestamp():
    return '[' + datetime.datetime.now().strftime("%H:%M:%S") + ']'

