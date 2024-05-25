# ZenGQ-Training

This repository contains the code and resources to train the `ZenGQ` model using the `Rep00Zon` dataset on Google Colab.

## Tree Structure
```tree
ZenGQ-Training/
    ├── README.md
    └── train_colab.ipynb
```

## Model

[ZenGQ](https://huggingface.co/prabinpanta0/ZenGQ)
ZenGQ is a small fine-tuned BERT model for question-answering tasks, trained on a custom dataset.


- **Model:** BERT-base-uncased
- **Task:** Question Answering
- **Dataset:** [Rep00Zon](https://huggingface.co/datasets/prabinpanta0/Rep00Zon)
- **license:** mit
- **datasets:** `prabinpanta0/Rep00Zon`
- **language:** English
- **metrics:** accuracy
- **pipeline_tag:** question-answering
- **tags:** general_knowledge, Question_Answers

---
## Dataset

[Rep00Zon](https://huggingface.co/datasets/prabinpanta0/Rep00Zon)

Rep00Zon is a small dataset designed for practicing question-answering tasks. It contains fewer than 1,000 question-context-answer pairs in English, providing a manageable size for beginners to work with.

- **Curated by:** Prabin Panta
- **Funded by :** N/A
- **Shared by :** Prabin Panta
- **Language(s) (NLP):** English
- **License:** MIT
- **task_categories:** question-answering
- **tags:** general_knowledge, Question_Answers
- **pretty_name:** `Rep00Zon`
- **size_categories:** n<1K

---
## Colab Training Code

You can use the provided Jupyter notebook to train the model on Google Colab. The notebook includes all the necessary steps to load the dataset, configure the training environment, and train the model.

[Open in Colab](https://colab.research.google.com/drive/1l66HQciYGEZoMswH-Z3DmJtCsbdynooa?usp=sharing)

## Installation

To run the code locally, you'll need to install the following packages:

```bash
pip install transformers accelerate datasets
```

##Usage
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/ZenGQ-Training.git`
2. Navigate to the project directory: `cd ZenGQ-Training`
3. Run the Jupyter notebook: `jupyter notebook train_colab.ipynb`


### **train_colab.ipynb**

Here's an example content for your `train_colab.ipynb` file:

```python
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Training ZenGQ Model\n",
        "\n",
        "This notebook provides the steps to train the ZenGQ model using the Rep00Zon dataset on Google Colab.\n",
        "\n",
        "## Install Dependencies\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install transformers accelerate datasets"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Load the Dataset and Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments\n",
        "from datasets import load_dataset\n",
        "\n",
        "# Load the dataset\n",
        "dataset = load_dataset(\"prabinpanta0/Rep00Zon\")\n",
        "\n",
        "# Load the tokenizer and model\n",
        "tokenizer = AutoTokenizer.from_pretrained(\"prabinpanta0/ZenGQ\")\n",
        "model = AutoModelForQuestionAnswering.from_pretrained(\"prabinpanta0/ZenGQ\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Training the Model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Define training arguments\n",
        "training_args = TrainingArguments(\n",
        "    output_dir='./results',\n",
        "    num_train_epochs=3,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=16,\n",
        "    warmup_steps=500,\n",
        "    weight_decay=0.01,\n",
        "    logging_dir='./logs',\n",
        ")\n",
        "\n",
        "# Create the Trainer\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=dataset['train'],\n",
        "    eval_dataset=dataset['validation']\n",
        ")\n",
        "\n",
        "# Train the model\n",
        "trainer.train()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Save the Model\n",
        "\n",
        "After training, save the model and tokenizer:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Save the model and tokenizer\n",
        "model.save_pretrained('./ZenGQ')\n",
        "tokenizer.save_pretrained('./ZenGQ')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Evaluate the Model\n",
        "\n",
        "Use the trained model for question answering tasks:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from transformers import pipeline\n",
        "\n",
        "# Load the model and tokenizer\n",
        "tokenizer = AutoTokenizer.from_pretrained('./ZenGQ')\n",
        "model = AutoModelForQuestionAnswering.from_pretrained('./ZenGQ')\n",
        "\n",
        "# Create a QA pipeline\n",
        "qa_pipeline = pipeline('question-answering', model=model, tokenizer=tokenizer)\n",
        "\n",
        "# Define a context and a question\n",
        "context = \"Berlin is the capital of Germany. Paris is the capital of France. Madrid is the capital of Spain.\"\n",
        "question = \"What is the capital of Germany?\"\n",
        "\n",
        "# Get the answer\n",
        "result = qa_pipeline(question=question, context=context)\n",
        "print(f\"Question: {question}\")\n",
        "print(f\"Answer: {result['answer']}\")"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```
