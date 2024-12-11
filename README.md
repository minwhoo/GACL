# Official Code and Data for "Target-Agnostic Gender-Aware Contrastive Learning for Mitigating Bias in Multilingual Machine Translation" (EMNLP 2023)

Official repo for "[Target-Agnostic Gender-Aware Contrastive Learning for Mitigating Bias in Multilingual Machine Translation](https://aclanthology.org/2023.emnlp-main.1046/)".

## Fine-tune NMT model with GACL

1. Install python dependencies
```bash
pip install -r requirements.txt
```

2. Get MT-GenEval dataset for evaluation

```bash
git clone https://github.com/amazon-science/machine-translation-gender-eval
```

1. Run training
```bash
bash run_translation.sh
```

## Citation

```bibtex
@inproceedings{lee-etal-2023-target,
    title = "Target-Agnostic Gender-Aware Contrastive Learning for Mitigating Bias in Multilingual Machine Translation",
    author = "Lee, Minwoo and Koh, Hyukhun and Lee, Kang-il and Zhang, Dongdong and Kim, Minsung and Jung, Kyomin",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    pages = "16825--16839",
}
```
