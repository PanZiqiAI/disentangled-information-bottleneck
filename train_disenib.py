
from methods.disenib_fc.config import ConfigTrain
from methods.disenib_fc.dataloader import generate_data
from methods.disenib_fc.disenib import DisenIB


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    model = DisenIB(cfg=cfg)
    dataloader = generate_data(cfg)
    # 3. Train
    model.train_parameters(**dataloader)
