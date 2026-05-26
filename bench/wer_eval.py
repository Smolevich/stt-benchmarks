import re
import sys

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) using Levenshtein distance at the word level.
    """
    def normalize(text):
        # Convert to lowercase and remove punctuation
        text = text.lower().replace("-", " ")
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    # Build DP matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance at the character level.
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if not ref_chars:
        return 1.0 if hyp_chars else 0.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 wer_eval.py \"<reference ground truth text>\" \"<hypothesis transcribed text>\"")
        sys.exit(1)

    ref = sys.argv[1]
    hyp = sys.argv[2]

    wer_val = compute_wer(ref, hyp)
    cer_val = compute_cer(ref, hyp)

    print(f"Reference:  \"{ref}\"")
    print(f"Hypothesis: \"{hyp}\"")
    print("-" * 40)
    print(f"Word Error Rate (WER):      {wer_val:.2%}  ({wer_val:.4f})")
    print(f"Character Error Rate (CER): {cer_val:.2%}  ({cer_val:.4f})")
