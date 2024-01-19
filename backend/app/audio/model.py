import datetime

class Audio:
    def __init__(self, length, name):
        self.length = length
        self.name = name
        self.created_date = datetime.datetime.now()

    def __str__(self):
        return f"Audio: {self.name}, Length: {self.length}, Created Date: {self.created_date}"
