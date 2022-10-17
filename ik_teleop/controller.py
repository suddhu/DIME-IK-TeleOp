from multiprocessing import Process

from ik_teleop.teleop_utils.dexarm_operation import DexArmOp
from omegaconf import DictConfig
import hydra 
from hydra.utils import get_original_cwd

import os

def robot_controller(cfg : DictConfig):
    control = DexArmOp(allegro_bound_path = os.path.join(get_original_cwd(), "bound_data", "allegro_bounds.yaml"), cfg = cfg)
    control.move()

@hydra.main(config_path="parameters", config_name="allegro_config")
def main(cfg: DictConfig) -> None:
    print("\n***************************************************************\n     Starting controller process \n***************************************************************")
    controller_process = Process(target = robot_controller, args=(cfg,))
    controller_process.start()
    print("\nController process started!\n")    
    controller_process.join()

if __name__ == '__main__':
    main()
