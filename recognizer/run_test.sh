rm pred_logs/test_predict_seq.$1.log
export CUDA_VISIBLE_DEVICES=0
python3 test.py $1
./test.sh $1
