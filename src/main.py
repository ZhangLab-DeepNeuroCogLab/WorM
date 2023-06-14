import os
import wandb
import torch

from src.args import build_parser
from src.model import WM_Model
from src.train import Trainer
from src.utils.logger import get_logger
from src.utils.data_utils import get_multitask_dataloader

def main():

    parser = build_parser()
    config = parser.parse_args()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:'+str(config.gpu) 
                          if torch.cuda.is_available() else 'cpu')
    
    if config.stage == 'Train':
        exp_name = 'WM_Bench'
        config.exp_name = exp_name

        if config.resume:
            wandb.init(
                project='WM_Bench',
                entity='', 
                config={}, 
                id=config.resume_wandb_id, 
                resume='must', 
                allow_val_change=True
            )
        else:
            wandb.init(
                project='WM_Bench',
                entity='', 
                config={}
            )

        assert wandb.run is not None
        config.run_name = wandb.run.name

        config.log_path = os.path.join(config.log_folder, config.run_name)
        config.model_path = os.path.join(config.model_folder, config.run_name)
        config.output_path = os.path.join(config.output_folder, config.run_name)

        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path, exist_ok=True)
        if not os.path.exists(config.model_path):
            os.makedirs(config.model_path, exist_ok=True)
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path, exist_ok=True)

        log_file_path = os.path.join(config.log_path, 'log.txt')
        logger = get_logger(config.run_name, log_file_path)

        multitask_dataloaders = get_multitask_dataloader(config)

        if config.gen_test:
            train_dataloader, val_dataloader, test_dataloader, gen_test_dataloader = multitask_dataloaders
        else:
            gen_test_dataloader = None
            train_dataloader, val_dataloader, test_dataloader = multitask_dataloaders

        model = WM_Model(config, device)
        model.to(device)

        logger.info('Model: {}'.format(model))
        logger.info('Config: {}'.format(config))
        wandb.config.update(config)

        trainer = Trainer(
            config, model, 
            train_dataloader, val_dataloader, test_dataloader, gen_test_dataloader, 
            wandb, device, logger
        )

        trainer.train()
        trainer.test()

        wandb.finish()

if __name__ == '__main__':
    main()