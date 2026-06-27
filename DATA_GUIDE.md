# Agricultural Training Data Guide

## Overview

AgriNexus AI includes a curated agricultural Q&A dataset designed for training domain-specific language models focused on farming and crop management.

## Dataset Structure

### Agricultural Q&A Dataset (`data/agricultural_qa_dataset.json`)

```json
{
  "metadata": {
    "name": "AgriNexus Agricultural QA Dataset",
    "version": "1.0.0",
    "description": "Agricultural question-answer pairs...",
    "total_qa_pairs": 30
  },
  "qa_pairs": [
    {
      "id": 1,
      "question": "How do I improve soil fertility?",
      "answer": "Improve soil fertility through crop rotation..."
    },
    ...
  ]
}
```

## Dataset Statistics

- **Total Q&A Pairs**: 30
- **Average Question Length**: 7.0 words
- **Average Answer Length**: 17.2 words
- **Domain Coverage**:
  - Soil management and fertility
  - Crop rotation and intercropping
  - Irrigation and water management
  - Pest and disease management
  - Fertilization and nutrients
  - Composting and organic matter
  - Seasonal planting and harvesting

## Data Categories

### 1. Soil Management (7 pairs)
- Soil fertility improvement
- Soil testing and pH management
- Soil erosion prevention
- Water retention in sandy soil

### 2. Crop Management (8 pairs)
- Crop rotation and monoculture
- Companion planting
- Intercropping
- Planting time and harvest timing

### 3. Irrigation & Water (3 pairs)
- Watering frequency
- Water retention
- Irrigation systems

### 4. Pest & Disease Control (3 pairs)
- Disease prevention
- Pest management
- Natural pest control

### 5. Nutrients & Fertilization (4 pairs)
- NPK nutrients
- Fertilizer selection
- Nutrient deficiency identification
- Fertilization frequency

### 6. Organic & Sustainable Practices (3 pairs)
- Organic farming
- Composting
- Cover crops

### 7. Garden Preparation (2 pairs)
- Soil preparation
- Garden startup

## Using the Dataset

### Load Dataset in Python

```python
from src.data_loader import load_qa_dataset, get_dataset_stats

# Load QA pairs
qa_pairs = load_qa_dataset("data/agricultural_qa_dataset.json")

# Get statistics
stats = get_dataset_stats("data/agricultural_qa_dataset.json")
print(stats)
```

### Create Data Loaders

```python
from src.data_loader import create_agricultural_dataloader
from src.tokenizer import CharTokenizer

tokenizer = CharTokenizer()
train_loader, val_loader = create_agricultural_dataloader(
    dataset_path="data/agricultural_qa_dataset.json",
    tokenizer=tokenizer,
    batch_size=8,
    max_length=128,
    train_split=0.8,
)
```

### Extract Knowledge Base

```python
from src.data_loader import create_knowledge_base_from_dataset
from src.memory import MemoryStore

facts = create_knowledge_base_from_dataset("data/agricultural_qa_dataset.json")
memory_store = MemoryStore(embedding_dim=256)
memory_store.add_facts(facts, source="agricultural_dataset")
memory_store.build_embeddings()
```

## Training on Agricultural Data

### Using the Agricultural Training Script

```bash
# Basic training
python train_on_agri_data.py --epochs 10 --batch-size 8

# With custom parameters
python train_on_agri_data.py \
  --epochs 15 \
  --batch-size 16 \
  --learning-rate 0.0005 \
  --d-model 256 \
  --num-layers 4 \
  --num-heads 8 \
  --max-length 128
```

### Training Output

- **Best Model**: `checkpoints/best_agri_model.pt`
- **Final Model**: `checkpoints/final_agri_model.pt`
- **Tokenizer**: `checkpoints/agri_tokenizer.json`

## Extending the Dataset

### Adding New Q&A Pairs

1. Edit `data/agricultural_qa_dataset.json`
2. Add new entry to `qa_pairs` array:

```json
{
  "id": 31,
  "question": "What is your new question?",
  "answer": "Your detailed agricultural answer here."
}
```

3. Update `metadata.total_qa_pairs`

### Best Practices for New Data

- **Questions**: Keep concise, 5-10 words
- **Answers**: Provide detailed, actionable information
- **Domain Accuracy**: Ensure information is agriculturally sound
- **Variety**: Cover different aspects and crops
- **Language**: Use clear, accessible language for farmers

## Data Format Specifications

### Question Guidelines
- Length: 5-15 words typical
- Structure: Direct question format
- Focus: Practical, common farming issues
- Example: "How do I improve soil fertility?"

### Answer Guidelines
- Length: 15-50 words typical
- Structure: Direct, actionable advice
- Include: Key practices, reasons why
- Format: Sentence fragments acceptable for clarity

## Performance Metrics

When training on this dataset:

```
Dataset Statistics:
- Total Q&A pairs: 30
- Train/Val split: 80/20
- Sequence length: 128 tokens
- Vocabulary size: 99 tokens

Typical Training Results (3 epochs, batch_size=4):
- Epoch 1: Train Loss: 1.16, Val Loss: 0.17
- Epoch 2: Train Loss: 0.14, Val Loss: 0.07
- Epoch 3: Train Loss: 0.05, Val Loss: 0.05
```

## Data Splits

Default split is 80% training / 20% validation:
- **Training samples**: ~24 QA pairs
- **Validation samples**: ~6 QA pairs

## Future Data Expansion

Recommended additions for production:
1. Crop-specific Q&A (wheat, corn, rice, vegetables)
2. Regional climate considerations
3. Seasonal farming calendars
4. Pest identification guides
5. Equipment and machinery information
6. Market and economic data
7. Government policies and regulations
8. Advanced soil testing protocols

## Data License

This dataset is provided for research and educational purposes.

## Integration with API

Once trained, use the model with the FastAPI server:

```bash
# Start API with agricultural model
python -m uvicorn app.api:app --port 8000

# Make inference requests
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I improve soil fertility?"}'
```

## Troubleshooting

### Dataset Not Found
```
Error: Dataset not found at data/agricultural_qa_dataset.json
Solution: Ensure the file exists in the data directory
```

### Low Training Loss
```
Problem: Loss stays high or doesn't improve
Solutions:
- Increase epochs
- Adjust learning rate
- Check data quality
- Verify batch size
```

### Memory Issues
```
Problem: Out of memory during training
Solutions:
- Reduce batch size
- Reduce max_length
- Use simpler model (fewer layers)
```

## References

- Agricultural practices based on USDA guidelines
- Crop management from university extension services
- Sustainable farming best practices
