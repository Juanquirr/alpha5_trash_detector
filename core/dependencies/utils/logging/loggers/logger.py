from abc import abstractmethod


class Logger:
    @abstractmethod
    def info(self, message):
        ...
    @abstractmethod
    def warning(self, message):
        ...
    @abstractmethod
    def error(self, message):
        ...