import spacy
import re
from gliner2 import GLiNER2

# Initialize models
nlp = spacy.load("en_core_web_sm")
gliner = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Simple KG storage
knowledge_graph = {"entities": [], "relations": []}
canonicalization_cache = {}

# Entity labels to extract
ENTITY_LABELS = [
    "person", "organization", "location", "product", "money", "date", "event", "dataset", "law", "legal_case", 
    "government_body", "legal_concept", "role"
]

def hard_prune(text):
    """Remove noise using regex"""
    text = re.sub(r'http\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Special chars
    return text.strip()

def rule_based_validation(entity, entity_type):
    """Validate entities based on rules"""
    if len(entity) < 2:  # Too short
        return False
    if entity_type == "person" and not entity[0].isupper():  # Person should be capitalized
        return False
    if entity_type == "money" and not any(c.isdigit() for c in entity):  # Money should have digits
        return False
    return True

def canonicalize(entity, entity_type):
    """Normalize entity name"""
    key = (entity.lower(), entity_type)
    if key in canonicalization_cache:
        return canonicalization_cache[key]
    
    # Simple normalization
    canonical = entity.strip().title() if entity_type == "person" else entity.strip()
    canonicalization_cache[key] = canonical
    return canonical

def write_to_kg(entity, entity_type):
    """Write to knowledge graph"""
    kg_entry = {"entity": entity, "type": entity_type}
    if kg_entry not in knowledge_graph["entities"]:
        knowledge_graph["entities"].append(kg_entry)

def pipeline(text):
    """Main pipeline"""
    print(f"\n{'='*60}")
    print(f"INPUT: {text}")
    print(f"{'='*60}")
    
    # 1. spaCy processing
    doc = nlp(text)
    print(f"\n1. SPACY - Tokens & POS:")
    for token in doc[:10]:  # Show first 10 tokens
        print(f"   {token.text:15} {token.pos_}")
    
    # 2. Hard pruning
    pruned_text = hard_prune(text)
    print(f"\n2. PRUNED TEXT: {pruned_text}")
    
    # 3. GLiNER2 extraction
    raw_result = gliner.extract_entities(pruned_text, ENTITY_LABELS)
    
    # Extract from nested dict structure: {'entities': {'label': ['text', ...]}}
    entities = []
    entity_dict = raw_result.get('entities', {})
    for label, entity_list in entity_dict.items():
        for entity_text in entity_list:
            entities.append({"text": entity_text, "label": label, "score": 1.0})
    
    print(f"\n3. GLINER2 - Extracted {len(entities)} entities:")
    for ent in entities:
        print(f"   {ent['text']:20} → {ent['label']:15} (conf: {ent['score']:.3f})")
    
    # 4. Rule-based validation
    validated = [e for e in entities if rule_based_validation(e['text'], e['label'])]
    print(f"\n4. VALIDATED: {len(validated)}/{len(entities)} entities passed")
    
    # 5. Canonicalization
    canonical_entities = []
    for ent in validated:
        canonical = canonicalize(ent['text'], ent['label'])
        canonical_entities.append({"entity": canonical, "type": ent['label'], "score": ent['score']})
        print(f"   {ent['text']:20} → {canonical}")
    
    # 6. KG write
    for ent in canonical_entities:
        write_to_kg(ent['entity'], ent['type'])
    print(f"\n6. KG UPDATED - Total entities in KG: {len(knowledge_graph['entities'])}")
    
    return canonical_entities


# Test the pipeline
if __name__ == "__main__":
    test_texts = [
        "Apple Inc. is buying a UK startup for $1 billion. Tim Cook announced this yesterday.",
        "Tesla's CEO Elon Musk visited the Berlin factory last week."
    ]
    
    for text in test_texts:
        pipeline(text)
    
    # Show final KG
    print(f"\n{'='*60}")
    print("FINAL KNOWLEDGE GRAPH:")
    print(f"{'='*60}")
    for entity in knowledge_graph["entities"]:
        print(f"  {entity['entity']:25} ({entity['type']})")
