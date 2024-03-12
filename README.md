# Evaluating Webcam-based Gaze Data as an Alternative for Human Rationale Annotations  

This repository contains code to the paper [Evaluating Webcam-based Gaze Data as an Alternative for Human Rationale Annotations](https://arxiv.org/pdf/2402.19133.pdf) accepted to LREC-COLING 2024.

Please refer to the paper for further details.

### 1 Fine-tuning models and extracting relevance scores/attention
In order to fine-tune the models on XQuAD (excl. the sentences from the eye-tracking corpus). 
You can run `finetuning.py` followed by `extract_lrp_relevance.py` and `extract_attention.py`. We recommend running this on a GPU.  

For instance like this, where `id` refers to a chosen integer to separate multiple runs with the same parameters.
```shell
model=roberta-base
language=en

python finetuning.py --training_languages ${language} --model ${model} --id ${SLURM_ARRAY_TASK_ID}
python extract_lrp_relevance.py --model ${model} --lang ${language} --id ${SLURM_ARRAY_TASK_ID} --case gi
python extract_lrp_relevance.py --model ${model} --lang ${language} --id ${SLURM_ARRAY_TASK_ID} --case lrp
python extract_attention.py --modelname ${model} --lang ${language} --id ${SLURM_ARRAY_TASK_ID}
```

### 2 Extracting reading patterns from WebQAmGaze
Total reading times can be extracted by running [this](https://github.com/tfnribeiro/WebQAmGaze/blob/main/scripts/meco_comparison/collect_relative_fixations.py) script
The following parameters need to be set inline depending on the analysis:
```python
MECO = False
WORKERS = "mturk_only"  # "mturk_only", "volunteer_only" or None (= "all")
FILTER_QUALITY = False
threshold = None
FILTER_CORRECT_ANSWER = False
FILTER_VISION = False
vision = 'normal' #"glasses", we only have this information for the data recorded at KU
```  

TBC
