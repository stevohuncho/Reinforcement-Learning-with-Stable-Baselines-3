from datetime import datetime
import os

def tf_logger(prefix: str = "") -> str: 
    log_path = f'../logs/{prefix}_{datetime.now().strftime("[%m-%d-%Y]_[%H-%M-%S]")}'
    os.makedirs(log_path, exist_ok=True)
    return log_path