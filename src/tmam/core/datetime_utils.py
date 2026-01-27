import datetime as dt

def serialize_datetime(value: dt.datetime) -> str:
    return value.isoformat()
