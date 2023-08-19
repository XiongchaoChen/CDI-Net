python test.py \
--resume './outputs/train/checkpoints/model_49.pt' \
--experiment_name 'test_49' \
--model_type 'model_cnn' \
--data_root '../../Data/Processed_7_2_AmapReconProj/' \
--norm 'BN' \
--net_filter 32 \
--n_denselayer 6 \
--growth_rate 32 \
--batch_size 2 \
--gpu_ids 0


