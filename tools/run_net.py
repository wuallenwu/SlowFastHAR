#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import torch.distributed as dist
import os
import subprocess
import random
import shutil
    
def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
        
        random.seed(0)

        # Perform training.
        if cfg.TRAIN.ENABLE:
            if not cfg.DATA.SYN_GEN:
                launch_job(cfg=cfg, init_method=args.init_method, func=train)
            else:
                rounds = 3
                num_epochs_per_round = cfg.SOLVER.MAX_EPOCH
                base_dir = '/home/emilykim/Desktop/HAR-project/REMAG_new/'
                seqs = os.listdir(base_dir)
                seqs = [s for s in seqs if int(s.split('-')[1].split('+')[-1]) in cfg.DATA.TRAIN_IDS]
                seqs.sort()
                dest_dir = cfg.DATA.PATH_TO_DATA_DIR_SYN
                os.makedirs(dest_dir, exist_ok=True)
                # for seq in seqs:
                #     if os.path.exists(base_dir + f'/{seq}/') != 0:
                #         os.symlink(base_dir + f'/{seq}', dest_dir + f'/{seq}', target_is_directory=True)
                #         continue
                    
                # launch_job(cfg=cfg, init_method=args.init_method, func=train)
                
                base_dir = '/home/emilykim/Desktop/HAR-project/extracted/SynCG-MC_Aerial_11Act_224px/Sequences/'
                seqs = os.listdir(base_dir)
                seqs = [s for s in seqs if int(s.split('-')[3].split('+')[-1]) not in cfg.DATA.TEST_IDS]
                # seqs.sort()
                random.shuffle(seqs)
                with open('remag_train_order_testset1.txt', 'w') as f:
                    for s in seqs:
                        f.write(s + '\n')
                        
                epochs = [30, 10, 5]
                percentage = [0.20, 0.50, 1.0]
                lr = [3e-3, 1e-3, 1e-3]
                
                for round in range(0, rounds):
                    if round >= 0:
                        
                        # epochs = [10, 10, 10]
                        
                        # epochs = [3, 3, 3, 3, 3, 3, 3]
                        # num_seqs = len(seqs) // rounds
                        end = int(len(seqs) * percentage[round])
                        if round == 0:
                            start = 0
                        else:
                            start = int(len(seqs) * percentage[max(round-1, 0)])
                        
                        for seq in seqs[start:end]:
                            os.symlink(base_dir + f'/{seq}', dest_dir + f'/{seq}', target_is_directory=True)
                            continue
                            
                            # image_folder = img_dir + img_seq + '/real/'
                            # images = os.listdir(image_folder)
                            # images.sort()
                            # image = image_folder + images[0]
                            
                            # subprocess.run(['bash', 'inference.sh', 'LHM-MINI', f'{image}', f'{base_dir}/{seq}/{seq.split(".")[0]}/smplx_params/'], cwd="LHM",)
                            # # if os.path.exists(dest_dir + f'/{seq}'):
                            # #     os.system(f'rm -r -f {dest_dir}/{seq}')
                            # if os.path.exists(f'{base_dir}/{seq}/{seq.split(".")[0]}/smplx_params/output/'):
                            #     os.makedirs(dest_dir_all + f'/{seq}-Round{round:03d}/', exist_ok=True)                                    
                            #     files = os.listdir(f'{base_dir}/{seq}/{seq.split(".")[0]}/smplx_params/output/')
                            #     for file in files:
                            #         shutil.move(f'{base_dir}/{seq}/{seq.split(".")[0]}/smplx_params/output/{file}', dest_dir_all + f'/{seq}-Round{round:03d}/')
                            #     # os.makedirs(dest_dir + f'/{seq}/')
                            #     os.symlink(dest_dir_all + f'/{seq}-Round{round:03d}', dest_dir + f'/{seq}-Round{round:03d}', target_is_directory=True)
                            #     if os.path.exists(dest_dir_all + f'/{seq}'):
                            #         shutil.copy(f'../REMAG_new/{seq}/labels.json', dest_dir_all + f'/{seq}-Round{round:03d}/labels.json')
                        if round >= 0:
                            if os.path.exists(cfg.OUTPUT_DIR + f'/checkpoints/checkpoint_epoch_{sum(epochs[:round]):05d}.pkl'): # cfg.SOLVER.MAX_EPOCH + 
                                cfg.TRAIN.CHECKPOINT_FILE_PATH = cfg.OUTPUT_DIR + f'/checkpoints/checkpoint_epoch_{sum(epochs[:round]):05d}.pkl'
                                cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
                            cfg.TRAIN.AUTO_RESUME = True
                            
                            cfg.SOLVER.MAX_EPOCH = sum(epochs[:round+1]) # cfg.SOLVER.MAX_EPOCH + 
                            cfg.SOLVER.BASE_LR = lr[round]
                        
                    launch_job(cfg=cfg, init_method=args.init_method, func=train)
                    
                base_dir = '/home/emilykim/Desktop/HAR-project/REMAG_new/'
                seqs = os.listdir(base_dir)
                seqs = [s for s in seqs if int(s.split('-')[1].split('+')[-1]) in cfg.DATA.TRAIN_IDS]
                seqs.sort()
                dest_dir = cfg.DATA.PATH_TO_DATA_DIR_SYN
                for seq in seqs:
                    if os.path.exists(base_dir + f'/{seq}/') != 0:
                        os.symlink(base_dir + f'/{seq}', dest_dir + f'/{seq}', target_is_directory=True)
                        continue
                cfg.SOLVER.BASE_LR = 1e-4
                cfg.TRAIN.CHECKPOINT_FILE_PATH = cfg.OUTPUT_DIR + f'/checkpoints/checkpoint_epoch_{sum(epochs[:round]):05d}.pkl'
                cfg.TRAIN.CHECKPOINT_TYPE = 'pytorch'
                cfg.SOLVER.MAX_EPOCH = sum(epochs[:round+1] + 10)
                
                launch_job(cfg=cfg, init_method=args.init_method, func=train)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
                num_view_list = [1, 3, 5, 7, 10]
                for num_view in num_view_list:
                    cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            else:
                launch_job(cfg=cfg, init_method=args.init_method, func=test)

        # Perform model visualization.
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo.
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    if dist.is_initialized():
        dist.barrier()
    main()
