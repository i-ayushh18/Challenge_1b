import json
import sys
from collections import defaultdict

# Usage: python map_to_example_format.py input.json output.json
if len(sys.argv) != 3:
    print("Usage: python map_to_example_format.py <input_json> <output_json>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Map metadata
metadata = data.get('metadata', {})
example_metadata = {
    'input_documents': metadata.get('input_documents', []),
    'persona': metadata.get('persona', ''),
    'job_to_be_done': metadata.get('job_to_be_done', ''),
    'processing_timestamp': metadata.get('processing_timestamp', '')
}

all_sections = data.get('extracted_sections', [])
# Sort by importance_rank (lower rank is more important)
sorted_sections = sorted(all_sections, key=lambda x: x.get('importance_rank', float('inf')))
# Take top 5 most important sections
top_sections = sorted_sections[:5]

# Map to output format
extracted_sections = []
for section in top_sections:
    mapped = {
        'document': section.get('document', ''),
        'section_title': section.get('section_title', ''),
        'importance_rank': section.get('importance_rank', 0),
        'page_number': section.get('page', section.get('page_number', 0))
    }
    extracted_sections.append(mapped)

# Get all subsections and sort by document and position
all_subsections = []
for sub in data.get('subsection_analysis', []):
    sort_key = (
        sub.get('document', ''),
        sub.get('page', sub.get('page_number', 0)),
        sub.get('refined_text', '')[:100]  # First 100 chars for stable sorting
    )
    all_subsections.append((sort_key, sub))

# Sort subsections by document name and page number
all_subsections.sort(key=lambda x: x[0])

# Take top 5 subsections overall
subsection_analysis = []
for _, sub in all_subsections[:5]:
    mapped = {
        'document': sub.get('document', ''),
        'refined_text': sub.get('refined_text', ''),
        'page_number': sub.get('page', sub.get('page_number', 0))
    }
    subsection_analysis.append(mapped)

# Compose mapped output
mapped_output = {
    'metadata': example_metadata,
    'extracted_sections': extracted_sections,
    'subsection_analysis': subsection_analysis
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(mapped_output, f, ensure_ascii=False, indent=4)

print(f"Mapped output written to {output_path}") 