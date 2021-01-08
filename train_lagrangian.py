
from methods.lagrangian_fc.config import ConfigTrain
from methods.lagrangian_fc.dataloader import generate_data
from methods.lagrangian_fc.ib_lagrangians import IBLagrangianModel


if __name__ == '__main__':
    # 1. Generate config
    cfg = ConfigTrain()
    # 2. Generate model & dataloader
    model = IBLagrangianModel(cfg=cfg)
    dataloader = generate_data(cfg)
    # 3. Train
    model.train_parameters(**dataloader)
