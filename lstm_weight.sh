is_lstm=1
drop=0.7
for d in 1 2 3 5 10 # 2  3  5 #1 #2 3 4 5 6 #1 2 3 4 5 6 7 8 9 10 #50 50 50 50 100 100 100 100  200 200 200 200 #5 10 20 50 100 200 500 1000 5000 #20 100 200 500 1000 #0.2 0.4 0.6 0.8 1 2 4 8
do
for w in 10 50 100 # 25 50 100 #20 50 100 #50 100 200 #500
do
for l in 1 2 3 #1 # 2 5 #2 3 4 #2 3 4
do
fn=intime_drop_${drop}_lstm_hidden_${w}_time_${d}_layer_${l}_islstm${is_lstm}.log
echo $fn
c##python run.py --model CommRNN --training_steps 1000 --batch_size 128 --n_hidden_dim 100 --experiment san-disk --oversample_pos --take_last_k_cycles $w --reg_coef 0 --concat_all_cycles --display_step 50 --val_display_step 990 --dropout_rate 0.0 --num_test_samples 0 --neg_weights 1 >& $fn
##python run.py --model CommRNN --training_steps 1000 --batch_size 128 --n_hidden_dim $w --experiment san-disk --oversample_pos --take_last_k_cycles $d --reg_coef 0 --display_step 50 --val_display_step 990 --dropout_rate 0.0 --num_test_samples 10000 --neg_weights 1 >& $fn
python run.py --concat_all_cycles  --model LSTM --training_steps 5000 --batch_size 128 --n_hidden_dim $w --experiment san-disk --oversample_pos --take_last_k_cycles $d --reg_coef 0 --display_step 50 --val_display_step 4950 --dropout_rate ${drop} --num_test_samples 0 --neg_weights 1 --n_layers $l --is_lstm ${is_lstm}  >& $fn
grep "with false pos" $fn -A 3 | tail -n1
done
done
done
