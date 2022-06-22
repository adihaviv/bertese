# BERTese

This repository contains the code for the models discussed in the paper "[BERTese: Learning to speck to BERT](https://arxiv.org/abs/2103.05327)".

Our code, both pretraining and training, is based on PyTorch (1.4.0) and 
Transformers (2.5.0). Note that all the dependencies and requirement file are provided in 
and [requirements.txt](requirements.txt).  

## Data

### Downloading the LAMA Dataset provided by [Petroni et al. 2019](https://arxiv.org/abs/1909.01066)

```bash
from https://github.com/facebookresearch/LAMA Download: 
curl -L https://dl.fbaipublicfiles.com/LAMA/data.zip
```
Note we are looking at the T-REx subset. 

### Downloading the Training set provided by [Jiang et al. 2020](https://arxiv.org/abs/1911.12543)

```bash
from https://github.com/jzbjyb/LPAQA Download: 
curl -L https://github.com/jzbjyb/LPAQA/blob/master/TREx_train.tar.gz
```

## Command for pretraining  
After downloading the dataset, you should first run the pretraining model by using the 
once the model is trained
```bash
python seq2seq_experiment.py \ 
    --output_dir <OUT_DIR> \
    --model_type bert_emb_identity_seq2seq \
    --model_name bert-large-uncased  \  
    --num_train_epochs 100 \  
    --evaluate_during_training \  
    --do_eval_test
``` 

##### Command for BERTese
```bash
python bertese_experiment.py \
    --log_examples 
    --evaluate_during_training 
    --model_type bertese 
    --train_batch_size 64 
    --max_seq_length 20 
    --do_train 
    --explicit_mask_loss_weight 0 
    --optimize_mask_softmin 
    --lpaqa 
    --num_train_epochs 20 
    --evaluate_during_training 
    --do_eval_dev 
```
    
To train with automatic mixed-precision, install [apex](https://github.com/NVIDIA/apex/) and add the ```--fp16``` flag.

## Citation

If you find this work helpful, please cite us
```
@misc{haviv2021bertese,
      title={BERTese: Learning to Speak to BERT}, 
      author={Adi Haviv and Jonathan Berant and Amir Globerson},
      year={2021},
      eprint={2103.05327},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This code is still improving. for any questions, please email adi.haviv@cs.tau.ac.il

