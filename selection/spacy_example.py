import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO."

# Process the text
doc = nlp(text)

# Print tokens and their attributes
print("Tokens and POS tags:")
for token in doc:
    print(f"{token.text:15} {token.pos_:10} {token.dep_:10}")

print("\n" + "="*50 + "\n")

# Named Entity Recognition
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text:20} {ent.label_:15} {spacy.explain(ent.label_)}")

print("\n" + "="*50 + "\n")

# Noun chunks
print("Noun Chunks:")
for chunk in doc.noun_chunks:
    print(f"{chunk.text:30} {chunk.root.text:15} {chunk.root.dep_}")
