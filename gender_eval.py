import string

import evaluate
import numpy as np
from datasets import Dataset


STRIP_PUNCT = str.maketrans(string.punctuation, ' '*len(string.punctuation))


class MTGenEvalEvaluator:
    def __init__(self, split, target_lang):
        self.split = split
        self.target_lang = target_lang
        self.predictions = []
        self.bleu_evaluator = evaluate.load("sacrebleu")

        # load data from disk
        self.src_sents = self._load_data("masculine", "en") + self._load_data("feminine", "en")
        self.tgt_sents = {
            "masculine": self._load_data("masculine", self.target_lang),
            "feminine": self._load_data("feminine", self.target_lang)
        }

        # process target data for evaluation
        self.tgt_tokens = {}
        for gender_type, d in self.tgt_sents.items():
            self.tgt_tokens[gender_type] = [ self._get_word_set(sent) for sent in d ]

        self.tgt_gender_dependent_tokens = {
            "masculine": [],
            "feminine": [],
        }
        for male_tokens, female_tokens in zip(self.tgt_tokens["masculine"], self.tgt_tokens["feminine"]):
            self.tgt_gender_dependent_tokens["masculine"].append(male_tokens - female_tokens)
            self.tgt_gender_dependent_tokens["feminine"].append(female_tokens - male_tokens)

    def _load_data(self, data_type, lang):
        path = f"./machine-translation-gender-eval/data/sentences/{self.split}/geneval-sentences-{data_type}-{self.split}.en_{self.target_lang}.{lang}"
        data = []
        with open(path) as f:
            for line in f:
                data.append(line.strip())
        return data

    def get_dataset(self):
        formatted_data = {"translation": [{"en": sent, "target_lang": self.target_lang} for sent in self.src_sents]}
        return Dataset.from_dict(formatted_data)

    def add_batch(self, predictions, references):
        """Function for metric compatibility"""
        self.predictions.extend(predictions)

    def compute(self):
        """Function for metric compatibility"""
        score = self.evaluate(self.predictions)
        self.predictions = []
        return score

    @staticmethod
    def _get_word_set(sent):
        return set(sent.lower().translate(STRIP_PUNCT).strip().split())

    def evaluate(self, preds, return_gender_predictions=False):
        pred_tokens = [ self._get_word_set(sent) for sent in preds ]
        male_preds = pred_tokens[:len(self.tgt_sents["masculine"])]
        female_preds = pred_tokens[len(self.tgt_sents["masculine"]):]
        assert len(male_preds) == len(self.tgt_sents["masculine"])
        assert len(female_preds) == len(self.tgt_sents["feminine"])

        male_score = [(len(m) == 0 or len(p & m) > 0) and len(p & f) == 0 for p, m, f in zip(male_preds, self.tgt_gender_dependent_tokens["masculine"], self.tgt_gender_dependent_tokens["feminine"])]
        female_score = [len(p & m) == 0 and (len(p & f) > 0 or len(f) == 0) for p, m, f in zip(female_preds, self.tgt_gender_dependent_tokens["masculine"], self.tgt_gender_dependent_tokens["feminine"])]
        male_score_simple = [len(p & f) == 0 for p, m, f in zip(male_preds, self.tgt_gender_dependent_tokens["masculine"], self.tgt_gender_dependent_tokens["feminine"])]
        female_score_simple = [len(p & m) == 0 for p, m, f in zip(female_preds, self.tgt_gender_dependent_tokens["masculine"], self.tgt_gender_dependent_tokens["feminine"])]
        combined_score = [all((s1, s2)) for s1, s2 in zip(male_score, female_score)]
        combined_score_simple = [all((s1, s2)) for s1, s2 in zip(male_score_simple, female_score_simple)]
        male_bleu_score = self.bleu_evaluator.compute(references=[[x] for x in self.tgt_sents["masculine"]], predictions=preds[:len(self.tgt_sents["masculine"])])["score"]
        female_bleu_score = self.bleu_evaluator.compute(references=[[x] for x in self.tgt_sents["feminine"]], predictions=preds[len(self.tgt_sents["masculine"]):])["score"]
        bleu_score = (male_bleu_score + female_bleu_score) / 2.0

        res = {
            "accuracy": np.mean(combined_score),
            "accuracy_male": np.mean(male_score),
            "accuracy_female": np.mean(female_score),
            "delta_g": np.mean(male_score) - np.mean(female_score),
            "accuracy_simple": np.mean(combined_score_simple),
            "accuracy_simple_male": np.mean(male_score_simple),
            "accuracy_simple_female": np.mean(female_score_simple),
            "delta_g_simple": np.mean(male_score_simple) - np.mean(female_score_simple),
            "bleu": bleu_score,
            "bleu_male": male_bleu_score,
            "bleu_female": female_bleu_score,
        }
        if return_gender_predictions:
            return res, combined_score
        return res
