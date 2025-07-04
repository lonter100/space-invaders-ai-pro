import logging

def setup_logger(logfile='log_ai.txt'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(logfile, 'a', 'utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__) 