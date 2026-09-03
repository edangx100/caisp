# Setup (uv virtual environment)

TextAttack pulls in a large dependency tree, so it must live in its own venv.

```bash
cd chp_2/g-performing-sentiment-analysis-using-an-llm

# 1. Create the environment (Python 3.10)
uv venv --python 3.10 .venv

# 2. Install everything.
#    The override is required: textattack -> flair 0.11.3 pins
#    sentencepiece==0.1.95, which has no wheel for Python 3.10 and fails to
#    build from source. Forcing a newer sentencepiece resolves it.
printf 'sentencepiece>=0.2.0\n' > overrides.txt
VIRTUAL_ENV=.venv uv pip install --overrides overrides.txt -r requirements.txt

# 3. Run
.venv/bin/python simple_sentiment_analyser_copy.py     # interactive analyser
.venv/bin/python run_attack.py                         # attack, textfooler
.venv/bin/python run_attack.py pwws                    # attack, WordNet synonyms
.venv/bin/python run_attack.py deepwordbug             # attack, character typos
```

`run_attack.py` downloads the NLTK language data it needs on first run
(`averaged_perceptron_tagger_eng`, `wordnet`, `stopwords`, ...). The
`textfooler` recipe additionally downloads the counter-fitted word embeddings
and the Universal Sentence Encoder the first time it runs, which takes a while.
