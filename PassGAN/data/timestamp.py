import datetime

UTC_OFFSET = 2

def timestamp():
    return '[' + (datetime.datetime.now()+datetime.timedelta(hours=UTC_OFFSET)).strftime("%H:%M:%S") + ']'

