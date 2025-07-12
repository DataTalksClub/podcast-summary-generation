from rouge_score import rouge_scorer

def evaluate_summary(original, summary):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(original, summary)['rougeL'].fmeasure
    return {
        "ROUGE-L F1": round(rouge_l, 3)
    }
