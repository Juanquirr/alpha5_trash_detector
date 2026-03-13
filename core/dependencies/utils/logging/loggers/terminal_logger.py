from core.dependencies.utils.logging.loggers.logger import Logger


class TerminalLogger(Logger):
    def info(self, message):
        print("Info: ", message)

    def warning(self, message):
        print("Warning: ", message)

    def error(self, message):
        print("Error: ", message)
