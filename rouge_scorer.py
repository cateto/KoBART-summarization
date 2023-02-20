import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge_metric import Rouge
import fastparquet
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from kobart import get_kobart_tokenizer
from tqdm import tqdm

class RougeScorer:
    def __init__(self):
        self.rouge_evaluator = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1024,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )

    def compute_rouge(self, ref_df, hyp_df):

        reference_summaries = ref_df.summary
        generated_summaries = hyp_df.summary
        

        scores = self.rouge_evaluator.get_scores(generated_summaries, reference_summaries)
        str_scores = self.format_rouge_scores(scores)
        #self.save_rouge_scores(str_scores)
        return str_scores

    def save_rouge_scores(self, str_scores):
        with open("rouge_scores.txt", "w") as output:
            output.write(str_scores)

    def format_rouge_scores(self, scores):
    	return "rouge-1 : {:.3f}, rouge-2 : {:.3f}, rouge-l : {:.3f}".format(
            scores["rouge-1"]["f"],
            scores["rouge-2"]["f"],
            scores["rouge-l"]["f"],
        )
     

def inference(docs:pd.DataFrame):
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary').cuda()
    tokenizer = get_kobart_tokenizer()
    
    target = docs.news.tolist()
    result = [summary(tokenizer, model, t) for t in tqdm(target)]
    df_output = pd.DataFrame({'news': target, 'summary': result})
    return df_output
    
def summary(tokenizer, model, text):
    max_length = 1024
    try:
        text = text.replace('\n', ' ')
        input_ids = tokenizer.encode(text)
        input_ids = add_padding_data(max_length, input_ids, tokenizer)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids.cuda(), eos_token_id=1, max_length=max_length, num_beams=4)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        return output
    except:
        print(text)
        print('=======================위의 텍스트에서 오류발생================')
        raise Exception()

def add_padding_data(max_length, inputs, tokenizer):
    if len(inputs) < max_length:
        pad = np.array([tokenizer.pad_token_id] *(max_length - len(inputs)))
        inputs = np.concatenate([inputs, pad])
    else:
        inputs = inputs[:max_length]
    return inputs

if __name__=='__main__':
    scorer = RougeScorer()
    test_file_path = '/home/woohyun/som/git/KoBART-summarization/data/pq/test.gz.parquet'
    df_test = fastparquet.ParquetFile(test_file_path).to_pandas()
    df_output = inference(df_test)
    score = scorer.compute_rouge(df_test, df_output) # df_test: test.tsv의 Dataframe, df_output: 같은 형식의 Dataframe
    print(score)