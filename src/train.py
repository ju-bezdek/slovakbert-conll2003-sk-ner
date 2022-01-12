import re
import os
import argparse
from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoConfig
from datasets import Dataset, ClassLabel, Sequence, Features, Value, load_dataset, load_metric
import numpy as np


def main(dataset_name: str, split:str , task:str, model_name: str, num_train_epochs,batch_size: int , revision:str) :
    
    if model_name == "/restore":
        model_name = "./cache/" + max(d for d in os.listdir("./cache") if d.startswith("checkpoint"))

    print(f"model:{model_name}")    
    print(f"dataset:{dataset_name}")    
    
    datasetDict = load_dataset(dataset_name)
    dataset = datasetDict[split]
    feature = dataset.features["ner_tags"].feature
    labels_count = feature.num_classes


    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    args = TrainingArguments(
        output_dir="./outputs",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=1e-5,
        save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
        load_best_model_at_end=True,
    )


    metric = load_metric("seqeval")

    label_list = feature.names
    
    #label_to_id = {i: i for i in range(len(label_list))}
    label_to_id = {label:i for i, label in  enumerate(label_list)}

    def tokenize_and_align_labels(tokenizer,examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels=[]
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Tokenize all texts and align the labels with them.
    """
    def tokenize_and_align_labels(tokenizer,examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            #padding=padding,
            truncation=True,
            #max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    #if data_args.label_all_tokens:
                    #label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    #else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    """
    
    #Now tokenize and align the labels over the entire dataset with Datasets map function:
    tokenized_dateset = dataset.map(lambda data: tokenize_and_align_labels(tokenizer,data), batched=True)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    tokenized_test_dateset = datasetDict["test"].map(lambda data: tokenize_and_align_labels(tokenizer,data), batched=True)
    tokenized_val_dateset = datasetDict["validation"].map(lambda data: tokenize_and_align_labels(tokenizer,data), batched=True)
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.


    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=labels_count,
        label2id= label_to_id,
        id2label= {i: l for l, i in label_to_id.items()},
        finetuning_task="ner",
        cache_dir="./cache",
        revision=revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    )
    print(config)
    model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            #num_labels=labels_count,
            config=config,
            revision=revision
            )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dateset if task=="train" else None,
        eval_dataset=tokenized_test_dateset if task=="train" else tokenized_dateset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        
    )

    if task=="train":
        train_result = trainer.train()
  
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_dateset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model("./outputs") 
        trainer.save_state()
        metrics = trainer.evaluate(eval_dataset=tokenized_val_dateset)
    else:
        metrics = trainer.evaluate()
    
    
    metrics["eval_samples"] = len(tokenized_dateset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    
    
       
    kwargs = {"finetuned_from": model_name, "tasks": "token-classification"}
    kwargs["dataset_tags"] = dataset_name
    kwargs["dataset"] = dataset_name
    #kwargs["metrics"]  = metrics
    trainer.create_model_card(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train model on dataset')
    parser.add_argument('--model_name', default="gerulata/slovakbert")
    parser.add_argument('--dataset_name', help='dataset name', default="ju-bezdek/conll2003-SK-NER")
    parser.add_argument('--task', help='train|eval', default="train")
    parser.add_argument('--split', help='train|test|val', default="train")
    parser.add_argument('--num_train_epochs', default="15", type=int)
    parser.add_argument('--batch_size', default="16", type=int)
    parser.add_argument('--revision', default="main")
    args = parser.parse_args()
    
    main(**vars(args))
    #main()