# test_bart.py
print("Testing BART installation...")

try:
    import torch
    print("✅ PyTorch installed")
except:
    print("❌ PyTorch NOT installed - run: pip install torch")
    exit()

try:
    from transformers import BartTokenizer, BartForConditionalGeneration
    print("✅ Transformers installed")
except:
    print("❌ Transformers NOT installed - run: pip install transformers")
    exit()

print("\n📥 Downloading BART model (this may take 5-10 minutes first time)...")
print("Downloading distilbart-cnn-12-6 (~1.2 GB)...\n")

try:
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
    print("✅ Model downloaded successfully!\n")
    
    # Test summarization
    text = "Mitochondria are organelles found in eukaryotic cells. They produce ATP through oxidative phosphorylation and are known as the powerhouse of the cell. They have their own DNA and can replicate independently."
    
    print("Testing summarization...")
    inputs = tokenizer([text], truncation=True, max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=20, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print(f"\n✅ BART is working!\n")
    print(f"Original: {text}\n")
    print(f"Summary: {summary}\n")
    
except Exception as e:
    print(f"\n❌ Error: {e}\n")
    print("This might be due to:")
    print("- Slow internet connection during download")
    print("- Insufficient disk space")
    print("- Network restrictions")
