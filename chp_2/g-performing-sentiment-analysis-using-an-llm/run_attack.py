# ============================================================================
# run_attack.py
#
# Runs an adversarial attack against simple_sentiment_analyser_copy.py.
#
# THE IDEA
# --------
# The sentiment model is accurate on normal text. An adversarial attack asks:
# can we make small, meaning-preserving edits to a sentence - swap a word for
# a synonym, say - so that a human still reads it the same way, but the model
# now gives the OPPOSITE answer? Every sentence where that works is a hole in
# the model.
#
# HOW TEXTATTACK DOES IT
# ----------------------
# An attack "recipe" is a bundle of four parts:
#   1. Goal function   - what counts as winning (here: make the model wrong).
#   2. Search method   - which words to try changing, and in what order.
#   3. Transformation  - how to change a word (synonym, typo, ...).
#   4. Constraints     - rules that keep the new sentence a fair example
#                        (still means the same, same part of speech, etc).
#
# THE RECIPE USED HERE: TextFoolerJin2019
# ---------------------------------------
#   Goal function  - untargeted misclassification: any wrong answer wins.
#   Search method  - greedy word importance ranking. It first deletes each
#                    word in turn to see which ones the model leans on most,
#                    then attacks those words first.
#   Transformation - swap a word for a near neighbour in a counter-fitted
#                    word embedding space (a space built so that synonyms sit
#                    close together and antonyms do not).
#   Constraints    - the replacement must keep the same part of speech, and
#                    the whole sentence must stay close in meaning as judged
#                    by the Universal Sentence Encoder. These are what stop
#                    the attack from simply rewriting the sentence.
#
# USAGE
#   python run_attack.py
# ============================================================================

import nltk

from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult
from textattack.attack_recipes import TextFoolerJin2019

from textattack_wrapper import SentimentWrapper
from simple_sentiment_analyser_copy import predict_sentiment, labels


# ---------------------------------------------------------------------------
# STEP 0 - Make sure the English language data TextAttack relies on is present.
#
# The recipes need to know things like "which word is a noun" and "what are
# the synonyms of this word". That knowledge lives in NLTK data files that are
# downloaded once and cached in your home directory. Without them the attack
# stops with a LookupError partway through, so we fetch them up front.
# ---------------------------------------------------------------------------
NLTK_DATA = [
    "averaged_perceptron_tagger_eng",  # part-of-speech tagger (current name)
    "averaged_perceptron_tagger",      # same tagger, older name some code asks for
    "universal_tagset",                # simplified POS tags used by constraints
    "wordnet",                         # synonym dictionary, used by PWWS
    "omw-1.4",                         # WordNet's multilingual companion data
    "stopwords",                       # common words the attacks refuse to change
    "punkt",                           # sentence/word splitter
]

print("Checking NLTK language data...")
for package in NLTK_DATA:
    # quiet=True keeps it silent when the data is already cached.
    nltk.download(package, quiet=True)



# ---------------------------------------------------------------------------
# STEP 1 - Wrap the model we want to attack.
# ---------------------------------------------------------------------------
print("Initializing model wrapper...")
model_wrapper = SentimentWrapper()


# ---------------------------------------------------------------------------
# STEP 2 - Build the dataset of sentences to attack.
#
# Format: (text, label). The label is the CORRECT answer, as a number that
# matches the model's output columns:
#       0 = negative, 1 = neutral, 2 = positive
#
# We only use clearly positive and clearly negative examples. Neutral sits
# between the other two classes, so flipping it is too easy to be interesting.
#
# The model was trained on tweets, so the sentences are written in that style.
# ---------------------------------------------------------------------------
examples = [
    # --- positive (label 2) ---
    ("This movie is great and amazing!", 2),
    ("I really enjoyed watching this film.", 2),
    ("The acting was superb and the storyline kept me engaged throughout.", 2),
    ("This is one of the best films I have watched in recent years.", 2),
    ("I absolutely loved the cinematography and the musical score was fantastic.", 2),
    ("The movie was quite good and I found it entertaining.", 2),
    ("I think this product is nice and works well for my needs.", 2),
    ("This book is interesting and kept my attention throughout.", 2),
    ("The food was delicious and the prices seemed reasonable.", 2),
    ("Wonderful service today, the staff were friendly and helpful.", 2),

    # --- negative (label 0) ---
    ("This was a terrible waste of time.", 0),
    ("The worst movie I've ever seen.", 0),
    ("I found the plot confusing and the characters were poorly developed.", 0),
    ("The movie was disappointing and failed to meet my expectations.", 0),
    ("The film was boring and I struggled to stay awake during the entire screening.", 0),
    ("The movie was somewhat disappointing and felt a bit slow.", 0),
    ("I found the product mediocre and it didn't meet my expectations.", 0),
    ("The book was awful and I thought it was really boring.", 0),
    ("The food was horrible and the experience left me unimpressed.", 0),
    ("Awful customer service, nobody bothered to help me at all.", 0),
]

# Dataset() just holds those pairs. label_names is only used for pretty output.
dataset = Dataset(examples, label_names=labels)


# ---------------------------------------------------------------------------
# STEP 3 - Build the attack.
#
# .build() takes our wrapper and assembles the goal function, search method,
# transformation and constraints described at the top of this file.
#
# First run only: this downloads the counter-fitted word embeddings (~480MB)
# and the Universal Sentence Encoder. Both are cached, so later runs skip
# straight to the attack.
# ---------------------------------------------------------------------------
print("Creating TextFooler attack...")
attack = TextFoolerJin2019.build(model_wrapper)


# ---------------------------------------------------------------------------
# STEP 4 - Attack settings.
# ---------------------------------------------------------------------------
print("Setting up attack arguments...")
attack_args = AttackArgs(
    num_examples=len(examples),                       # attack every sentence
    log_to_csv="attack_results.csv",                  # full log for later review
    checkpoint_interval=5,                            # save progress every 5
    disable_stdout=False,                             # show each result live
    random_seed=42,                                   # reproducible runs
)


# ---------------------------------------------------------------------------
# STEP 5 - Run it. The Attacker feeds each example to the attack and collects
# a result object per sentence.
# ---------------------------------------------------------------------------
print("Starting attack...\n")
attacker = Attacker(attack, dataset, attack_args)
results = attacker.attack_dataset()


# ---------------------------------------------------------------------------
# STEP 6 - Our own review of the results.
#
# TextAttack already prints a summary table, but we re-run every original and
# every perturbed sentence through the analyser ourselves. That independently
# confirms the model really did change its mind - we never just take the
# attack framework's word for it.
# ---------------------------------------------------------------------------
print("\n\nCustom Attack Results Summary")
print("=" * 80)

successful = 0     # attacks that flipped the prediction
failed = 0         # attacks that could not flip it
skipped = 0        # sentences the model already got wrong before any attack
total_words_changed = 0
total_words = 0

for i, result in enumerate(results, 1):
    print(f"\nExample {i}")
    print("-" * 80)

    original_text = result.original_result.attacked_text.text
    original = predict_sentiment(original_text)

    print(f"Original text     : {original_text}")
    print(f"Original sentiment: {original['label']} (confidence: {original['confidence']:.4f})")

    # The model was already wrong here, so there was nothing to attack.
    if isinstance(result, SkippedAttackResult):
        print("- Skipped: the model was already wrong on the clean sentence.")
        skipped += 1
        print("-" * 80)
        continue

    if isinstance(result, SuccessfulAttackResult):
        perturbed_text = result.perturbed_result.attacked_text.text
        perturbed = predict_sentiment(perturbed_text)

        print(f"\nPerturbed text     : {perturbed_text}")
        print(f"Perturbed sentiment: {perturbed['label']} (confidence: {perturbed['confidence']:.4f})")

        if original['label'] != perturbed['label']:
            print(f"SUCCESS: sentiment flipped from {original['label']} to {perturbed['label']}!")
        else:
            # Should not happen - if it does, our wrapper and the dataset
            # labels are probably out of sync.
            print("WARNING: marked successful, but our own check shows no flip!")

        # How much of the sentence did the attack have to touch? Ask TextAttack
        # rather than comparing words ourselves: it tracks the edits exactly,
        # even when a change makes the two sentences different lengths.
        words_changed = len(result.original_result.attacked_text.all_words_diff(
            result.perturbed_result.attacked_text))
        word_count = len(result.original_result.attacked_text.words)

        successful += 1
        total_words_changed += words_changed
        total_words += word_count

        print(f"\nWords changed    : {words_changed} of {word_count}")
        print(f"Modification rate: {words_changed / word_count:.2%}")
    else:
        # FailedAttackResult: the search ran out of options without flipping it.
        print(f"\nAttack failed - could not move the model off '{original['label']}'.")
        failed += 1

    print("-" * 80)


# ---------------------------------------------------------------------------
# STEP 7 - The headline numbers.
# ---------------------------------------------------------------------------
attacked = successful + failed   # sentences the attack actually got to try

print("\n" + "=" * 80)
print("FINAL SCORE")
print("=" * 80)
print("Recipe used            : TextFoolerJin2019")
print(f"Sentences in dataset   : {len(results)}")
print(f"Skipped (already wrong): {skipped}")
print(f"Attacked               : {attacked}")
print(f"Successful attacks     : {successful}")
print(f"Failed attacks         : {failed}")

if attacked:
    # Success rate = of the sentences the model got RIGHT, how many did we
    # manage to break? This is the number that says how fragile the model is.
    print(f"Attack success rate    : {successful / attacked:.2%}")
if successful:
    print(f"Average words changed  : {total_words_changed / successful:.2f}")
    print(f"Average perturbation   : {total_words_changed / total_words:.2%} of each sentence")

print("\nFull log saved to: attack_results.csv")
