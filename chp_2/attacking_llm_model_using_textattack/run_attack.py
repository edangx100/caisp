import torch
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from textattack_wrapper import SentimentWrapper
from sentiment_classifier import predict_sentiment

# Initialize the model wrapper
print("Initializing model wrapper...")
model_wrapper = SentimentWrapper()


# Create a dataset of examples to attack
# Format: (text, label) where label is 0 for negative, 1 for positive
examples = [
    ("This movie is great and amazing!", 1),
    ("This was a terrible waste of time.", 0),
    ("I really enjoyed watching this film.", 1),
    ("The worst movie I've ever seen.", 0),
    ("The acting was superb and the storyline kept me engaged throughout.", 1),
    ("I found the plot confusing and the characters were poorly developed.", 0),
    ("This is one of the best films I have watched in recent years.", 1),
    ("The movie was disappointing and failed to meet my expectations.", 0),
    ("I absolutely loved the cinematography and the musical score was fantastic.", 1),
    ("The film was boring and I struggled to stay awake during the entire screening.", 0),
    ("The movie was quite good and I found it entertaining.", 1),
    ("I think this product is nice and works well for my needs.", 1),
    ("The service was acceptable and the staff were helpful.", 1),
    ("This book is interesting and kept my attention throughout.", 1),
    ("The food was decent and the prices seemed reasonable.", 1),
    # Negative examples (that could be attacked while preserving meaning)
    ("The movie was somewhat disappointing and felt a bit slow.", 0),
    ("I found the product mediocre and it didn't meet my expectations.", 0),
    ("The service was okay but nothing special and rather basic.", 0),
    ("The book was average and I thought it was somewhat boring.", 0),
    ("The food was passable but the experience left me unimpressed.", 0)
]

# Convert to TextAttack dataset format
dataset = Dataset(examples)


# Create the TextFooler attack
print("Creating attack...")
attack = TextFoolerJin2019.build(model_wrapper)

# Set up attack arguments
print("Setting up attack arguments...")
attack_args = AttackArgs(
    num_examples=20,  # Number of examples to attack
    log_to_csv="attack_results.csv",  # Save results to CSV file
    checkpoint_interval=5,  # Save a checkpoint every 5 examples
    disable_stdout=False  # Show output in terminal
)

# Start the attack
print("Starting attack...")
attacker = Attacker(attack, dataset, attack_args)
results = attacker.attack_dataset()





# Custom analysis of results
print("\nCustom Attack Results Summary:")
print("-" * 50)

# Initialize counters
successful = 0
failed = 0
total_words_changed = 0
total_words = 0

# Detailed analysis of each result
print("\nDetailed Results:")
print("=" * 80)
for i, result in enumerate(results, 1):
    print(f"\nExample {i}:")
    print("-" * 80)

    # Get the original text
    original_text = str(result.original_result.attacked_text)

    # Get original sentiment prediction
    original_sentiment = predict_sentiment(original_text)
    original_label = original_sentiment['label']
    original_conf = original_sentiment['confidence']

    print(f"Original text: {original_text}")
    print(f"Original sentiment: {original_label} (confidence: {original_conf:.4f})")

    # Check if attack was successful
    if isinstance(result, SuccessfulAttackResult):
        # Get the perturbed text
        perturbed_text = str(result.perturbed_result.attacked_text)

        # Get perturbed sentiment prediction
        perturbed_sentiment = predict_sentiment(perturbed_text)
        perturbed_label = perturbed_sentiment['label']
        perturbed_conf = perturbed_sentiment['confidence']

        print(f"\nPerturbed text: {perturbed_text}")
        print(f"Perturbed sentiment: {perturbed_label} (confidence: {perturbed_conf:.4f})")

        # Check if sentiment actually flipped
        sentiment_flipped = (original_label != perturbed_label)
        if sentiment_flipped:
            print(f"SUCCESS: Sentiment flipped from {original_label} to {perturbed_label}!")
        else:
            print(f"WARNING: Attack marked as successful but sentiment did not flip!")

        # Calculate word changes
        orig_words = original_text.split()
        pert_words = perturbed_text.split()

        # Simple word change calculation
        words_changed = sum(1 for o, p in zip(orig_words, pert_words) if o != p)
        if len(orig_words) != len(pert_words):
            words_changed += abs(len(orig_words) - len(pert_words))

        # Update counters
        successful += 1
        total_words_changed += words_changed
        total_words += len(orig_words)

        # Print statistics
        print(f"\nWords changed: {words_changed}")
        print(f"Modification rate: {words_changed/len(orig_words):.2%}")
    else:
        print(f"\n✗ Attack failed - Could not change sentiment from {original_label}")
        failed += 1

    print("-" * 80)



