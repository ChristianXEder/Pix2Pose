#!/bin/bash
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 1
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 5
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 6
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 8
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 9
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 10
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 11
DISPLAY=:0 python3 tools/3_train_pix2pose_wo_background_crop.py 0 cfg/cfg_bop2020_rgb_custom.json lmo_random_texture_all 12
