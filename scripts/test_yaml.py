from src.training.trainer_config_parser import ConfigParser

def test():
    parser = ConfigParser()
    config = parser.parse()
    print(config)


if __name__ == '__main__':
    test()
