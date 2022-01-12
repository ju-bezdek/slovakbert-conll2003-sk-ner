from transformers import AutoModel
import argparse

def main(model_path, target_name):
    model_path = "./outputs/result_model" 
    
    model = AutoModel.from_pretrained(model_path)
    print(f"Model on '{model_path}' config:")
    print(model.config)
    print()
    print(f"Push {model_path} to huggingface hub as {target_name}? [Y/N]")
    if input().upper()!="Y":
        print("Exiting...")
    model.push_to_hub(target_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Push model to Huggingface hub')
    parser.add_argument('--model_path', default="./outputs/result_model" )
    parser.add_argument('--target_name', default="ju-bezdek/slovakbert-conll2003-sk-ner")
    args = parser.parse_args()
    
    main(**vars(args))