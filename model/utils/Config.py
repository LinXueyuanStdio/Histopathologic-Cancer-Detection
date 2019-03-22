import json
from .general import init_dir, copyfile

class Config():
    """
    Class that loads hyperparameters from json file into attributes
    """

    def __init__(self, source):
        """
        Args:
            source: path to json file or dict
        """
        self.source = source

        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load_json(s)
        else:
            self.load_json(source)

    def load_json(self, source):
        with open(source) as f:
            data = json.load(f)
            self.__dict__.update(data)

    def save(self, dir_name):
        init_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            json.dumps(self.source, indent=4)
        else:
            copyfile(self.source, dir_name + self.export_name)

