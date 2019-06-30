# Example of running DeepSet for 'img-sum' experiment.
```console
edocohen@rack-gamir-g03:~$ python run.py \
                                --model DeepSet \
                                --training_steps 200000 \
                                --batch_size 16 \
                                --n_hidden_dim 100 \
                                --experiment img-sum \
                                --lr_schedule '[(1e4, 1e-3), (2e4, 5e-4), (5e4, 1e-4), (1e5, 1e-5), (1e6, 1e-6)]' \
                                --input_model_arch '[200, 200]' \
                                --output_model_arch '[100, 50]'
```
