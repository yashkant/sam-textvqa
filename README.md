Spatially Aware Multimodal Transformers for TextVQA
===================================================
We propose a novel spatially aware self-attention layer such that each visual entity only looks at neighboring entities defined by a spatial graph and use it to solve TextVQA.
<p align="center">
  <img src="tools/sam-textvqa-large.png">
</p>
Paper: https://arxiv.org/abs/2007.12146

## Repository Setup

Create a fresh conda environment, and install all dependencies.

```text
conda create -n spat python=3.6
conda activate spat
cd spat-textvqa
pip install -r requirements.txt
```

Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Finally, install apex from: https://github.com/NVIDIA/apex

## Data Setup
Download files from the dropbox [link](https://www.dropbox.com/sh/dk6oubjlt2x7w0h/AAAKExm33IKnVe8mkC4tOzUKa) and place it in the ``data/`` folder.
Ensure that data paths match the directory structure provided in ``data/README.md``

## Run Experiments
From the below table pick the suitable configuration file:

 | Method  |  context (c)   |  Train splits   |  Evaluation Splits  | Config File|
 | ------- | ------ | ------ | ------ | ------ |
 | SA-M4C  | 3 | TextVQA | TextVQA | train-tvqa-eval-tvqa-c3.yml |
 | SA-M4C  | 3 | TextVQA + STVQA | TextVQA | train-tvqa_stvqa-eval-tvqa-c3.yml |
 | SA-M4C  | 3 | STVQA | STVQA | train-stvqa-eval-stvqa-c3.yml |
 | SA-M4C  | 5 | TextVQA | TextVQA | train-tvqa-eval-tvqa-c5.yml |

To run the experiments use:
```
python train.py \
--config config.yml \
--tag experiment-name
```

To evaluate the pretrained checkpoint provided use:
```
python train.py \
--config configs/train-tvqa_stvqa-eval-tvqa-c3.yml \
--pretrained_eval data/pretrained-models/best_model.tar
```
Note: The beam-search evaluation is 
undergoing changes and will be updated.

## Citation
Cite this work as:
```
@article{Kant2020SpatiallyAM,
  title={Spatially Aware Multimodal Transformers for TextVQA},
  author={Yash Kant and Dhruv Batra and Peter Anderson and A. Schwing and D. Parikh and Jiasen Lu and Harsh Agrawal},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.12146}
}
```

## Acknowledgements
Parts of this codebase were borrowed from the following repositories:
- [12-in-1: Multi-Task Vision and Language Representation Learning](https://github.com/facebookresearch/vilbert-multi-task): Training Setup
- [MMF: A multimodal framework for vision and language research](https://github.com/facebookresearch/mmf/): Dataset processors and M4C model

We thank <a href="https://abhishekdas.com/">Abhishek Das</a> and <a href="https://amoudgl.github.io/">Abhinav Moudgil</a> for their feedback. 
The Georgia Tech effort was supported in part by NSF, AFRL, DARPA, ONR YIPs, ARO PECASE, Amazon. 
The views and conclusions contained herein are those of the authors and should not be interpreted
 as necessarily representing the official policies or endorsements, either expressed or implied, of the U.S. Government, or any sponsor.


## License
MIT
