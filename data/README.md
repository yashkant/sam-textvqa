## Data Organization
Organize the downloaded data files from [Dropbox link](https://www.dropbox.com/sh/dk6oubjlt2x7w0h/AAAKExm33IKnVe8mkC4tOzUKa) as per below structure:
```
data
|
├── textvqa/
│   ├── tvqa_trainval_obj.lmdb    # extracted object features from train-val         
│   ├── tvqa_trainval_ocr.lmdb    # extracted ocr features from train-val         
│   ├── tvqa_test_obj.lmdb    # extracted object features from test         
│   ├── tvqa_test_ocr.lmdb    # extracted ocr features from test         
│   ├── tvqa_test_imdb.npy    # dataset split         
│   ├── tvqa_train_imdb.npy    # dataset split         
│   └── tvqa_val_imdb.npy    # dataset split         
|
├── stvqa/
│   ├── stvqa_trainval_obj.lmdb    # extracted object features from train-val         
│   ├── stvqa_trainval_ocr.lmdb    # extracted ocr features from train-val         
│   ├── stvqa_test_obj.lmdb    # extracted object features from test         
│   ├── stvqa_test_ocr.lmdb    # extracted ocr features from test         
│   ├── tvqa_test_imdb.npy    # dataset split         
│   ├── tvqa_train_imdb.npy    # dataset split         
│   └── tvqa_val_imdb.npy    # dataset split         
|
├── vocabs/                         # different vocabs for stvqa and textvqa
│   ├── fixed_answer_vocab_stvqa_5k.txt
│   └── fixed_answer_vocab_textvqa_5k.txt
|
├── pretrained_models/best_model.tar        # pretrained model
|
└── evaluation/                     # required for evaluation 
    ├── stvqa_eval_df_test.pkl
    ├── stvqa_eval_df.pkl
    ├── tvqa_eval_df_test.pkl
    └── tvqa_eval_df.pkl
```
