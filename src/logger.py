from datetime import datetime
import os

class Logger():
    def __init__(self, name: str) -> None:
        self.dir = f'../logs/{name}_{datetime.now().strftime("[%m-%d-%Y]_[%H-%M-%S]")}'
        os.makedirs(self.dir, exist_ok=True)

    def tf_logger(self, prefix: str = "") -> str: 
        log_path = f'{self.dir}/{prefix}'
        os.makedirs(log_path, exist_ok=True)
        return log_path
    
    def get_dir(self) -> str:
        return self.dir