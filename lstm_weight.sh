is_lstm=0
for d in 2 #1 #2 3 4 5 6 #1 2 3 4 5 6 7 8 9 10 #50 50 50 50 100 100 100 100  200 200 200 200 #5 10 20 50 100 200 500 1000 5000 #20 100 200 500 1000 #0.2 0.4 0.6 0.8 1 2 4 8
do
for w in 10 #20 50 100 #50 100 200 #500
do
for l in 1 #2 3 4
do
fn=nodrop5_lstm_hidden_${w}_time_${d}_layer_${l}_islstm${is_lstm}.log
echo $fn
##python run.py --model CommRNN --training_steps 1000 --batch_size 128 --n_hidden_dim 100 --experiment san-disk --oversample_pos --take_last_k_cycles $w --reg_coef 0 --concat_all_cycles --display_step 50 --val_display_step 990 --dropout_rate 0.0 --num_test_samples 0 --neg_weights 1 >& $fn
##python run.py --model CommRNN --training_steps 1000 --batch_size 128 --n_hidden_dim $w --experiment san-disk --oversample_pos --take_last_k_cycles $d --reg_coef 0 --display_step 50 --val_display_step 990 --dropout_rate 0.0 --num_test_samples 10000 --neg_weights 1 >& $fn
python run.py --concat_all_cycles --model LSTM --training_steps 5000 --batch_size 128 --n_hidden_dim $w --experiment san-disk --oversample_pos --take_last_k_cycles $d --reg_coef 0 --display_step 50 --val_display_step 4900 --dropout_rate 0.0 --num_test_samples 0 --neg_weights 1 --n_layers $l --is_lstm ${is_lstm} >& $fn
grep "with false pos" $fn -A 3 | tail -n1
done
done
done
