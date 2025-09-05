#!/bin/bash
export PYTHONPATH=/ws
python /ws/src/main_enhanced.py +experiment=acid_enhancer data_loader.train.batch_size=8 data_loader.train.num_workers=8 output_dir=EXP_SAVING_PATH