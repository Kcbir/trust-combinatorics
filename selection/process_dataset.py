import spacy
import re
import json
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

def pipeline(text, max_text_length=2000):
    """Main pipeline with text truncation for large inputs"""
    # Truncate very long texts
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."
    
    print(f"\n{'='*60}")
    print(f"INPUT (first 500 chars): {text[:500]}...")
    print(f"{'='*60}")
    
    # 1. spaCy processing
    doc = nlp(text)
    print(f"\n1. SPACY - Tokens & POS (first 10):")
    for token in list(doc)[:10]:
        print(f"   {token.text:15} {token.pos_}")
    
    # 2. Hard pruning
    pruned_text = hard_prune(text)
    print(f"\n2. PRUNED TEXT (first 300 chars): {pruned_text[:300]}...")
    
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
        print(f"   {ent['text']:30} → {ent['label']:15} (conf: {ent['score']:.3f})")
    
    # 4. Rule-based validation
    validated = [e for e in entities if rule_based_validation(e['text'], e['label'])]
    print(f"\n4. VALIDATED: {len(validated)}/{len(entities)} entities passed")
    
    # 5. Canonicalization
    canonical_entities = []
    for ent in validated:
        canonical = canonicalize(ent['text'], ent['label'])
        canonical_entities.append({"entity": canonical, "type": ent['label'], "score": ent['score']})
        print(f"   {ent['text']:30} → {canonical}")
    
    # 6. KG write
    for ent in canonical_entities:
        write_to_kg(ent['entity'], ent['type'])
    print(f"\n6. KG UPDATED - Total entities in KG: {len(knowledge_graph['entities'])}")
    
    return canonical_entities


def load_jsonl_entries(filepath, num_entries=2):
    """Load first N entries from a JSONL file"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_entries:
                break
            entries.append(json.loads(line.strip()))
    return entries


# Main execution
if __name__ == "__main__":
    dataset_path = "/Users/kabir/Desktop/Learing to Forget (CGP)/dataset/train.jsonl"
    
    print("="*70)
    print("PROCESSING TOP 2 ENTRIES FROM train.jsonl")
    print("="*70)
    
    # Load top 2 entries
    entries = load_jsonl_entries(dataset_path, num_entries=2)
    
    for idx, entry in enumerate(entries):
        print(f"\n\n{'#'*70}")
        print(f"ENTRY {idx + 1}: ID = {entry.get('id', 'N/A')}")
        print(f"{'#'*70}")
        
        # Get the input text (which contains the question + conversation)
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')
        
        print(f"\nQUESTION + TRANSCRIPT LENGTH: {len(input_text)} characters")
        print(f"EXPECTED OUTPUT: {output_text[:200]}...")
        
        # Run pipeline on the input
        results = pipeline(input_text)
        
        print(f"\n--- EXTRACTED ENTITIES FOR ENTRY {idx + 1} ---")
        for ent in results:
            print(f"  • {ent['entity']} ({ent['type']})")
    
    # Show final KG
    print(f"\n\n{'='*70}")
    print("FINAL KNOWLEDGE GRAPH (ALL ENTRIES COMBINED):")
    print(f"{'='*70}")
    for entity in knowledge_graph["entities"]:
        print(f"  {entity['entity']:35} ({entity['type']})")
    
    print(f"\nTotal unique entities: {len(knowledge_graph['entities'])}")
