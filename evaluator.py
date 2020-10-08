import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sam.datasets.metrics import STVQAANLSEvaluator, TextVQAAccuracyEvaluator
from sam.task_utils import forward_model
from tools.registry import registry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)

vqa_evaluator = TextVQAAccuracyEvaluator()
anls_evaluator = STVQAANLSEvaluator()


class Evaluator:
    def __init__(self, checkpoint_path, model, dataloaders, task):
        self.vocabs = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.dataloaders = dataloaders
        self.task = task
        self.n_gpu = torch.cuda.device_count()
        registry["is_running_validation"] = True
        self.restore_checkpoint()
        self.load_vocabs()

    def load_vocabs(self):
        for task, vocab_path in zip(
            ["textvqa", "stvqa"],
            [registry["Vocabs"]["vocab5k"], registry["Vocabs"]["vocab5k_stvqa"]],
        ):
            vocab = []
            with open(vocab_path) as f:
                for line in f.readlines():
                    vocab.append(line.strip())
            self.vocabs[task] = vocab

    def evaluate_no_beam(self, split):
        evalai_file = os.path.join(
            os.path.dirname(self.checkpoint_path),
            f"evalai_{split}.json",
        )

        evalai_preds = self.run_model_no_beam(split)

        with open(evalai_file, "w") as f:
            json.dump(evalai_preds, f)

        print(f"Dumping file: {evalai_file}")



    def evaluate(self, split, beam_size):
        eval_df_key = f"{registry['val_on'][0]}_{split}"
        eval_df = pd.read_pickle(registry["Evaluation"][eval_df_key])
        logger.info(f"Processing split: {split}")

        predictions = self.run_model(beam_size, split)
        predictions["complete_seqs"] = np.concatenate(
            [x.cpu().reshape(-1, 12) for x in predictions["complete_seqs"]], axis=0
        ).tolist()
        predictions["topkscores"] = np.concatenate(
            [x.cpu() for x in predictions["topkscores"]], axis=0
        ).tolist()
        predictions["question_id"] = np.concatenate(
            [x.cpu() for x in predictions["question_id"]], axis=0
        ).tolist()

        if "answers" not in eval_df:
            eval_df["answers"] = [["none"] * 10] * len((eval_df["question_id"]))

        # Compute VQA and ANLS accuracies
        results_df = pd.DataFrame.from_dict(predictions, orient="columns")
        accuracies_vqa = evaluate_predictions(
            eval_df, results_df, self.vocabs[self.task], acc_type="vqa"
        )
        accuracies_anls = evaluate_predictions(
            eval_df, results_df, self.vocabs[self.task], acc_type="anls"
        )

        # Log results on validation set
        if "test" not in split:
            logger.info(
                "{} Accuracy: {} for {} questions, split {}, dataset {}".format(
                    "vqa",
                    accuracies_vqa["vqa_accuracy"],
                    accuracies_vqa["accuracies_df"].shape,
                    split,
                    self.task,
                )
            )

            logger.info(
                "{} Accuracy: {} for {} questions, split {}, dataset {}".format(
                    "anls",
                    accuracies_anls["vqa_accuracy"],
                    accuracies_anls["accuracies_df"].shape,
                    split,
                    self.task,
                )
            )

        evalai_file = os.path.join(
            os.path.dirname(self.checkpoint_path),
            f"evalai_{split}_beam_{beam_size}.json",
        )

        # EvalAI/ST-VQA file
        answer_dict = []
        for i, pred in accuracies_vqa["best_result_df"].iterrows():
            answer_dict.append(
                {
                    "question_id": pred["question_id"],
                    "answer": pred["pred_answer"].strip(),
                }
            )

        with open(evalai_file, "w") as f:
            json.dump(answer_dict, f)

        print(f"Dumping file: {evalai_file}")

    def run_model(self, beam_size, split):
        # set beam-size
        self.model.module.set_beam_size(beam_size)

        predictions = {
            "question_id": [],
            "topkscores": [],
            "complete_seqs": [],
            # 'ocr_tokens': []
        }
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc="Beam Search Evaluation"):
                # Batch is updated inside the method, no outputs are needed
                forward_model(
                    None, self.device, self.model, batch_dict=batch, beam_search=True
                )
                save_keys = ["question_id", "topkscores", "complete_seqs"]
                for key in save_keys:
                    predictions[key].append(batch[key])
                break

        self.model.train()
        return predictions

    def run_model_no_beam(self, split):
        scores, batch_sizes = [], []
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch_dict in tqdm(self.dataloaders[split], desc=f"Eval on {split}"):
                loss, score, batch_size, batch_predictions = forward_model(
                    {"loss": "textvqa", "metric": "textvqa"}, self.device, self.model, batch_dict=batch_dict
                )
                scores.append(score * batch_size)
                batch_sizes.append(batch_size)
                predictions.extend(batch_predictions)

        evalai_preds = [{"question_id": x["question_id"], "answer": x["pred_answer"]} for x in predictions]
        return evalai_preds

    def restore_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint_dict = {}

        for attr in checkpoint["model_state_dict"]:
            if not attr.startswith("module."):
                checkpoint_dict[f"module.{attr}"] = checkpoint["model_state_dict"][attr]
            else:
                checkpoint_dict[attr] = checkpoint["model_state_dict"][attr]

        checkpoint_epoch = int(checkpoint["epoch_id"]) + 1
        logger.info(
            f"Restoring Checkpoint: {self.checkpoint_path}; Epoch: {checkpoint_epoch}"
        )
        self.model.load_state_dict(checkpoint_dict)


def vqa_calculate(batch_dict, vocab):
    pred_answers = batch_dict["pred_answers"]
    ocr_tokens_enc = batch_dict["ocr_tokens"]
    gt_answers_enc = batch_dict["answers"]
    topkscores = batch_dict["topkscores"]
    answer_space_size = len(vocab)

    predictions = []

    for idx, question_id in enumerate([batch_dict["question_id"]]):
        context_tokens = ocr_tokens_enc[idx]
        answer_words = []
        belongs_to = []

        for answer_id in pred_answers[idx].tolist():
            if answer_id >= answer_space_size:
                belongs_to.append("ocr")
                answer_id -= answer_space_size
                answer_words.append(context_tokens[answer_id])
            else:
                if answer_id == registry["EOS_IDX"]:
                    belongs_to.append("vocab+eos")
                    break
                belongs_to.append("vocab")
                answer_words.append(vocab[answer_id])

        answer = " ".join(answer_words).replace(" 's", "'s")
        gt_answers = gt_answers_enc[idx]

        predictions.append(
            {
                "question_id": question_id,
                "gt_answers": gt_answers,
                "pred_answer": answer,
                "belongs_to": belongs_to,
                "answer_words": answer_words,
                "topkscores": topkscores,
                "pred_ids": pred_answers,
            }
        )

    accuracy, pred_scores = vqa_evaluator.eval_pred_list(predictions)
    return {
        "question_id": predictions[0]["question_id"],
        "accuracy": accuracy,
        "pred_answer": predictions[0]["pred_answer"],
        "belongs_to": predictions[0]["belongs_to"],
        "answer_words": predictions[0]["answer_words"],
        "topkscores": predictions[0]["topkscores"],
    }


def anls_calculate(batch_dict, vocab):
    pred_answers = batch_dict["pred_answers"]
    ocr_tokens_enc = batch_dict["ocr_tokens"]
    gt_answers_enc = batch_dict["answers"]
    topkscores = batch_dict["topkscores"]
    answer_space_size = len(vocab)

    predictions = []
    for idx, question_id in enumerate([batch_dict["question_id"]]):
        context_tokens = ocr_tokens_enc[idx]
        answer_words = []
        belongs_to = []

        for answer_id in pred_answers[idx].tolist():
            if answer_id >= answer_space_size:
                belongs_to.append("ocr")
                answer_id -= answer_space_size
                answer_words.append(context_tokens[answer_id])
            else:
                if answer_id == registry["EOS_IDX"]:
                    belongs_to.append("vocab+eos")
                    break
                belongs_to.append("vocab")
                answer_words.append(vocab[answer_id])

        answer = " ".join(answer_words).replace(" 's", "'s")
        gt_answers = gt_answers_enc[idx]

        predictions.append(
            {
                "question_id": question_id,
                "gt_answers": gt_answers,
                "pred_answer": answer,
                "belongs_to": belongs_to,
                "answer_words": answer_words,
                "topkscores": topkscores,
                "pred_ids": pred_answers,
            }
        )

    try:
        accuracy, pred_scores = anls_evaluator.eval_pred_list(predictions)
    except:
        import pdb

        pdb.set_trace()

    return {
        "question_id": predictions[0]["question_id"],
        "accuracy": accuracy,
        "pred_answer": predictions[0]["pred_answer"],
        "belongs_to": predictions[0]["belongs_to"],
        "answer_words": predictions[0]["answer_words"],
        "topkscores": predictions[0]["topkscores"],
    }


def evaluate_predictions(eval_df, results_df, vocab, acc_type="vqa", tokens_from="vd"):
    if acc_type == "vqa":
        calculate = vqa_calculate
    elif acc_type == "anls":
        calculate = anls_calculate
    else:
        raise AssertionError

    predictions = []
    for i in range(results_df.shape[0]):
        re = results_df.iloc[i]
        question_id = re.question_id
        vd = eval_df[eval_df["question_id"] == question_id].iloc[0]

        tokens_key = "ocr_tokens_y"
        if tokens_key not in eval_df:
            tokens_key = "ocr_tokens"
            assert tokens_key in eval_df

        if tokens_from == "re":
            tokens = re[tokens_key]
        else:
            tokens = vd[tokens_key]

        batch = {
            "question_id": re.question_id,
            "answers": [vd.answers],
            "ocr_tokens": [tokens],
            "topkscores": [re.topkscores],
            "pred_answers": np.array([re.complete_seqs[1:]]),
        }

        calculate_result = calculate(batch, vocab)
        calculate_result["pred_ids"] = np.array([re.complete_seqs])

        predictions.append(calculate_result)

    accuracies_df = pd.DataFrame(predictions)
    best_result = []
    oracle_accuracies = 0.0
    for qid, row in accuracies_df.groupby("question_id"):
        idx = np.argmax(row.topkscores)
        # idx = np.random.randint(row.topkscores.shape[0])
        oracle_accuracies += row.accuracy.tolist()[idx]
        best_result.append(row.iloc[idx])

    best_result_df = pd.DataFrame(best_result)
    mean_acc = oracle_accuracies / accuracies_df["question_id"].unique().shape[0]
    return {
        "vqa_accuracy": mean_acc,
        "accuracies_df": accuracies_df,
        "best_result_df": best_result_df,
    }
