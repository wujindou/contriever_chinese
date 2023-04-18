from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel, BertModel
import json
import re


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class QuestionReferenceDensity_forPredict(torch.nn.Module):
    def __init__(self, question_encoder_path, reference_encoder_path) -> None:
        super().__init__()
        self.question_encoder = Contriever.from_pretrained(question_encoder_path)
        self.reference_encoder = Contriever.from_pretrained(reference_encoder_path)

       
    def forward(self, question, sentences):
        temp = 0.05
        cls_q = self.question_encoder(**question)
        cls_r_sentences = self.reference_encoder(**sentences)
        cls_q /= temp
        results = torch.matmul(cls_q, torch.transpose(cls_r_sentences, 0, 1))
        return results
    
class QuestionReferenceDensityScorer:
    def __init__(self, question_encoder_path, reference_encoder_path, device=None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(question_encoder_path)
        self.model = QuestionReferenceDensity_forPredict(question_encoder_path, reference_encoder_path)
        self.select_inputs = None
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if not device else device
        self.model = self.model.to(self.device).eval()

    def get_embeddings(self, sentences: list[str]) -> torch.Tensor:
        # Tokenization and Inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            question_inputs = self.tokenizer([sentences[0]], padding=True,
                                    truncation=True, return_tensors='pt')
           
            for key in question_inputs:
                question_inputs[key] = question_inputs[key].to(self.device)
            if not self.select_inputs:
                
                select_inputs = self.tokenizer(sentences[1:], padding=True,
                                    truncation=True, return_tensors='pt')
                for key in select_inputs:
                    select_inputs[key] = select_inputs[key].to(self.device)
                self.select_inputs = select_inputs
            outputs = self.model(question_inputs, self.select_inputs)
            sentence_embeddings = outputs

            return sentence_embeddings

    def score_documents_on_query(self, query: str, documents: list[str]) -> torch.Tensor:
        result = self.get_embeddings([query, *documents])
        return result[0]

    def select_topk(self, query: str, documents: list[str], k=1):
        """
        Returns:
            `ret`: `torch.return_types.topk`, use `ret.values` or `ret.indices` to get value or index tensor
        """
        scores = []
        max_batch = 256
        for i in range((len(documents) + max_batch - 1) // max_batch):
            scores.append(self.score_documents_on_query(query, documents[max_batch*i:max_batch*(i+1)]).to('cpu'))
        scores = torch.concat(scores)
        return scores.topk(min(k, len(scores)))

def test_contriever_scorer():
    import collections
    uids = []
    sentences=[]
    idx = 100000
    query_results = collections.defaultdict(list)
    idx_to_doc = {}
    with open('cross.train.demo.tsv','r',encoding='utf-8') as lines:
        for line in lines:
            data = line.strip().split('\t')
            query = data[0]
            passage = data[2]
            label = int(data[-1])
            idx+=1
            json_data = {'query':query,'passage':passage,'label':label,'idx':idx}
            query_results[query].append(json_data)
            idx_to_doc[idx]= json_data
            uids.append(idx)
            sentences.append(passage)
#     for d in json_data:sentences.extend(d['sentences'])
#     sentences = open('retrieval_data.txt').read().split('\n')
    checkpoint='facebook/mcontriever-msmarco'
    scorer = QuestionReferenceDensityScorer(checkpoint, checkpoint)
    total_cnt =0
    sum_score = 0
    import time
    for q in sorted(query_results.keys())[:20]:
        start = time.time()
        target_idx = scorer.select_topk(q, sentences, 5).indices
        top5_uids = [uids[t_id] for t_id in target_idx]
        print(target_idx)
        print(top5_uids)
        for d in query_results[q]:print(d)
        ids = [d['idx'] for d in query_results[q] if d['label']>=1]
        print(ids)
        recall_at_5 = len(set(ids)&(set(top5_uids)))/len(ids) if  len(ids)>0 else 0.0
        print(recall_at_5)
        sum_score+=recall_at_5
        total_cnt+=1
        print('cost ='+str(time.time()-start))
        sys.exit(1)
    print('recall @ 5 ='+str(float(sum_score)/total_cnt))

if __name__ == "__main__":
    test_contriever_scorer()
