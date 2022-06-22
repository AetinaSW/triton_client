from asyncore import write
import logging
from flask import Flask

def conf_for_flask(stream=True, file=True, level='debug', file_name='innotis-client.log', write_mdoe='a'):
    
    # console_conf = { 'console': {
    #                     'class': 'logging.StreamHandler',
    #                     'stream': 'ext://flask.logging.wsgi_errors_stream',
    #                     'formatter': 'default' }}

    # file_conf = { 'file': {
    #             'class': 'logging.FileHandler',
    #             'formatter': 'default',
    #             'filename': f'{file_name}',
    #             'encoding': 'utf-8' }}

    logger_conf = {
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
            'datefmt': "%m-%d %H:%M:%S"
        }},
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'default'
            },
            'file': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'filename': f'{file_name}',
                'mode': f'{write_mdoe}',
                'encoding': 'utf-8'
            }
        },
        'root': {
            'level': f'{level.upper()}',
            'handlers': ['console', 'file']
        }
    }
    
    return logger_conf

""" get flask logger """
def get_flask_logger(name):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    else:
        raise Exception('No such logger')  

class CustomLogger:

    def __init__(self, name, log_file="", write_mode="w"):
        
        self.name = name
        self.log_file = log_file
        self.write_mode = write_mode
        self.write_log = True if bool(self.log_file.rstrip()) else False
        self.formatter = logging.Formatter(
                        "%(asctime)s %(levelname)-.4s %(message)s",
                        "%m-%d %H:%M:%S")
        
    """ get logger """
    def get_logger(self):
        logger = logging.getLogger(self.name)
        return logger if logger.hasHandlers() else self.create_logger()  
    
    """ Create logger which name is 'dev', and write log into innotis-client.log """
    def create_logger(self):

        logger = logging.getLogger(self.name)
        
        # setup LEVEL
        logger.setLevel(logging.DEBUG)
        
        # setup formatter
        if self.formatter == None:
            self.formatter = logging.Formatter(
                            "%(asctime)s %(levelname)-.4s %(message)s",
                            "%m-%d %H:%M:%S")
            
        # setup handler
        stream_handler = logging.StreamHandler()
        if self.write_log: file_handler = logging.FileHandler(self.log_file, self.write_mode, 'utf-8')
        
        # add formatter into handler
        stream_handler.setFormatter(self.formatter)
        if self.write_log: file_handler.setFormatter(self.formatter)

        # add handler into logger
        logger.addHandler(stream_handler)
        if self.write_log: logger.addHandler(file_handler)

        logger.info('Create logger: {}'.format(self.name))
        logger.info('Write log file: {} ({})'.format(self.write_log, self.log_file))
        return logger

if __name__ == '__main__':

    from logging.config import dictConfig

    dictConfig(conf_for_flask(write_mdoe='a'))
    logger = logging.getLogger()

    logger.info('test')