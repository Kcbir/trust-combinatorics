from gliner2 import GLiNER2

model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = model.extract_entities(text, ["company", "person", "product", "location"])

print("Result type:", type(result))
print("Result:", result)
print("\nFormatted:")
for key, value in result.items() if isinstance(result, dict) else enumerate(result):
    print(f"{key}: {value}")
