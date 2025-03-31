# 🎬 Sentiment Prediction with the IMDB Dataset

This project performs binary sentiment classification (positive/negative) on the IMDB movie reviews dataset using a Transformer-based model from the Hugging Face 🤗 Transformers library.

## Features
- Fine-tunes a pre-trained transformer (e.g., BERT) on the IMDB dataset.
- Includes training, evaluation, and model saving.
- Custom metric tracking (e.g., accuracy, precision, recall, F1) via `metrics.py`.
- Modular code organized for easy experimentation.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
   Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Train the model**  
Run the training script:

```bash
python imdb_training.py
```

This will:
- Load and tokenize the IMDB dataset.
- Fine-tune a transformer model.
- Save the trained model and tokenizer to the `./imdb-sentiment` directory.

**Evaluate**  
Metrics such as accuracy, precision, recall, and F1 score are calculated during training and logged to the console. The evaluation logic is implemented in `metrics.py`.

## Project Structure
```bash
.
├── imdb_training.py     # Main script for training and evaluation
├── metrics.py           # Custom metrics functions
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── imdb-sentiment/      # Saved model directory (created after training)
```

## Model & Tokenizer
By default, the script uses a BERT-based model from Hugging Face's model hub. You can easily switch models by modifying the model_name in the script.

## Example Output
After training, your terminal will show evaluation metrics like:
```mathematica
Epoch 1 - Accuracy: 0.89 - Precision: 0.90 - Recall: 0.88 - F1: 0.89
```

## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
