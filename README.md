# Description
Traning procedure and evaluation for [ju-bezdek/slovakbert-conll2003-sk-ner](https://huggingface.co/ju-bezdek/slovakbert-conll2003-sk-ner)

## Training
For local training run 
>python src/train.py

For training on azure run

>python train_on_azure.py create_ws -sub_id <your azure subscription id> -ws <workspace name> -rg <resource group>

then

>python train_on_azure.py run_remote

## Trained model usage

```python:
from transformers import pipeline, AutoModel, AutoTokenizer
from spacy import displacy
import os


model_path="ju-bezdek/slovakbert-conll2003-sk-ner"

aggregation_strategy="max"
ner_pipeline = pipeline(task='ner', model=model_path, aggregation_strategy=aggregation_strategy)

input_sentence= "Ruský premiér Viktor Černomyrdin v piatok povedal, že prezident Boris Jeľcin , ktorý je na dovolenke mimo Moskvy , podporil mierový plán šéfa bezpečnosti Alexandra Lebedu pre Čečensko, uviedla tlačová agentúra Interfax"
ner_ents = ner_pipeline(input_sentence)
print(ner_ents)

ent_group_labels = [ner_pipeline.model.config.id2label[i][2:] for i in ner_pipeline.model.config.id2label if i>0]

options = {"ents":ent_group_labels}

dicplacy_ents = [{"start":ent["start"], "end":ent["end"], "label":ent["entity_group"]} for ent in ner_ents]
displacy.render({"text":input_sentence, "ents":dicplacy_ents}, style="ent", options=options, jupyter=True, manual=True)
```

### Result: 
<div>
             <span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
       <mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Ruský
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MISC</span>
       </mark>
        premiér 
       <mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Viktor Černomyrdin
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
       </mark>
        v piatok povedal, že prezident 
       <mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Boris Jeľcin,
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
       </mark>
        , ktorý je na dovolenke mimo 
       <mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Moskvy
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
       </mark>
        , podporil mierový plán šéfa bezpečnosti 
       <mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Alexandra Lebedu
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PER</span>
       </mark>
        pre 
       <mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Čečensko,
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">LOC</span>
       </mark>
        uviedla tlačová agentúra 
       <mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
           Interfax
           <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
       </mark>
       </div></span>
       </div>
