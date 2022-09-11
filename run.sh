############## TEST ################
#python demo.py --task FaceEnhancement --model GPEN-BFR-512 --in_size 512 --channel_multiplier 2 --narrow 1 --use_sr --sr_scale 2 --use_cuda --save_face --indir examples/imgs --outdir examples/outs-bfr
#python demo.py --task FaceEnhancement --model GPEN-BFR-256 --in_size 256 --channel_multiplier 1 --narrow 0.5 --use_sr --sr_scale 4 --use_cuda --save_face --indir examples/imgs --outdir examples/outs-bfr
python3 demo.py \
--alpha 0.8 \
--channel_multiplier 2 \
--ext .png \
--in_size 512 \
--indir input/ \
--model GPEN-BFR-512 \
--narrow 1 \
--outdir output/ \
--save_face \
--sr_model realesrnet \
--sr_scale 4 \
--task FaceEnhancement \
--tile_size 400 \
--use_cuda \
--use_sr



############## TRAIN ################
#CUDA_VISIBLE_DEVICES='0,1,' python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 train_simple.py --size 512 --channel_multiplier 2 --narrow 1 --ckpt ckpt-512 --sample sample-512 --batch 1 --path your_path_of_aligned_faces
