import math
import numpy as np

def BLEU_score(candidate, references):
	candidate_s = candidate.split()

	score = []
	for reference in references:
		precision = 0
		reference_s = reference.split()
		for w in candidate_s:
			if w in reference_s:
				precision += 1
		precision = precision/len(candidate_s) if len(candidate_s) > 0 else 0

		if len(candidate_s) > len(reference_s):
			bp = 1
		else:
			bp = np.exp(1-len(reference_s)/len(candidate_s)) if len(candidate_s) > 0 else 0

		bleu = precision * bp

		score.append(bleu)

	return np.mean(score)