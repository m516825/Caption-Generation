import math
import numpy as np

def BLEU_score(candidate, references):
	candidate_s = candidate.split()

	score = []
	for reference in references:
		precision = 0
		reference_s = reference.split()
		for w in reference_s:
			if w in candidate_s:
				precision += 1
		precision = precision/len(candidate_s)

		if len(candidate_s) > len(reference_s):
			bp = 1
		else:
			bp = np.exp(1-len(reference_s)/len(candidate_s))

		bleu = precision * bp

		score.append(bleu)

	return max(score)