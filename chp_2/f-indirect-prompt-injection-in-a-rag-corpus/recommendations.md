## Simple Defenses on the Retrieval Side
Hardening the system prompt helps, but it is not enough if poisoned files still reach the model. Below are plain-language defenses teams use on the document and RAG side.

1. Treat the knowledge base like code – Use merge requests, reviewers, and (where possible) signed or hashed snapshots so stray files cannot land without oversight.
2. Scan documents before indexing – Look for risky phrases (“ignore previous instructions”, fake roles, fake system blocks) or lines that look like secrets before you build embeddings.
3. Show sources to the user – Ask the assistant to cite which file or chunk it used. Flag answers that cite unknown or conflicting sources.
4. Run a second check on retrieved chunks – A small classifier or rules can run on what search returned, separate from checks on raw user typing.
5. Limit tools and side effects – If the assistant can call APIs or read email, make sure bad model output cannot trigger big actions without a human step.

No team rolls out all five on day one. The main lesson is: do not only guard the chat box, guard what gets read and retrieved as well.