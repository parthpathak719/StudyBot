from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- List of models your app uses ---
MODELS_TO_DOWNLOAD = [
    # The T5 model for Question Generation
    "mrm8488/t5-base-finetuned-question-generation-ap",
    
    # The BART model for Summarization
    "sshleifer/distilbart-cnn-12-6"
]

def download():
    for model_name in MODELS_TO_DOWNLOAD:
        print(f"--- Downloading: {model_name} ---")
        
        # Download tokenizer
        try:
            AutoTokenizer.from_pretrained(model_name)
            print(f"Tokenizer for {model_name} downloaded.")
        except Exception as e:
            print(f"Error downloading tokenizer for {model_name}: {e}")
            
        # Download model
        try:
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Model {model_name} downloaded.")
        except Exception as e:
            print(f"Error downloading model for {model_name}: {e}")
            
        print(f"--- Finished: {model_name} ---\n")

if __name__ == "__main__":
    print("Starting model download process...")
    download()
    print("All models have been downloaded to the cache.")
