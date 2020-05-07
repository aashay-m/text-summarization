from pathlib import Path
from rouge_score import rouge_scorer, scoring
import nltk
from nltk.translate.meteor_score import single_meteor_score
import argparse
import statistics

def calculate_scores(output_lns, reference_lns, score_path):
    score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    meteor = []

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)
        meteor.append(single_meteor_score(reference=reference_ln, hypothesis=output_ln))

    result = aggregator.aggregate()
    result["meteor"] = statistics.mean(meteor)
    score_file.write(
        "ROUGE_1: \n{} \n\n ROUGE_2: \n{} \n\n ROUGE_L: \n{} \n\n METEOR: \n{} \n\n".format(
            result["rouge1"], result["rouge2"], result["rougeL"], result["meteor"]
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Rouge Score Measures')

    parser.add_argument('--source', required=True, type=str, help='Generated Lines Location')
    parser.add_argument('--reference', required=True, type=str, help='Reference Lines Location')
    parser.add_argument('--score_path', required=True, type=str, help='Output Path')

    args = parser.parse_args()

    generated_lines = open(args.source, "r", encoding="utf-8").readlines()
    reference_lines = open(args.reference, "r", encoding="utf-8").readlines()

    calculate_scores(generated_lines, reference_lines, args.score_path)
