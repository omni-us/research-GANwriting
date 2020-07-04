python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_1.in_vocab_tr_writer ~/datasets/iam_final_forms/test_writer_real_imgs
python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_2.in_vocab_te_writer ~/datasets/iam_final_forms/train_writer_real_imgs
python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_3.oo_vocab_tr_writer ~/datasets/iam_final_forms/test_writer_real_imgs
python fid_score_crop64x64.py --gpu 0 --batch-size=256 res_4.oo_vocab_te_writer ~/datasets/iam_final_forms/train_writer_real_imgs

