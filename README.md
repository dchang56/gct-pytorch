# Graph Convolutional Transformer in Pytorch

I've reimplemented the original tensorflow implementation of [this paper](https://arxiv.org/pdf/1906.04716.pdf) by Choi et al, following materials provided.

The step-by-step instructions are pretty much the same as the original repo: https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer

I did not implement the synthetic parts of the paper, and I've left quite a bit of alternative model architecture and training details in the code (which I think is appropriate given the engineering-heavy and experimental nature of the project).

Here is an example of a bash script to run experiments. There are two possible tasks: mortality prediction and readmission prediction. Just assign either 'expired' or 'readmission' as the `LABEL_KEY`

```bash
export DATA_DIR='data dir'
export CUDA_VISIBLE_DEVICES="2"
LABEL_KEY=readmission

for LR in 1e-3 1.5e-3 2e-3; do
    for DROPOUT in 0.4 0.5 0.6 0.7; do
            OUTPUT_DIR='output dirs_${LR}_${DROPOUT}'
            mkdir -p $OUTPUT_DIR

            python train.py \
            --data_dir $DATA_DIR \
            --fold 50 \
            --output_dir $OUTPUT_DIR \
            --use_prior \
            --use_guide \
            --output_hidden_states \
            --output_attentions \
            --do_train \
            --do_eval \
            --do_test \
            --label_key $LABEL_KEY \
            --max_steps 1000000 \
            --hidden_dropout_prob $DROPOUT \
            --num_stacks 2 \
            --learning_rate $LR
        done
    done
done
```

## Comments

In my experience, training is quite unstable, and replicating the results exactly as presented in the paper has been a challenge. Still, this reimplementation was a decent learning experience.

