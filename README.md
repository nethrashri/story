cd ~/Desktop/Testing_local_gemma_workings
mkdir -p all-MiniLM-L6-v2/1_Pooling
cd all-MiniLM-L6-v2

# Download main files
curl -L -o config.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
curl -L -o model.safetensors "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"
curl -L -o tokenizer_config.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json"
curl -L -o tokenizer.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
curl -L -o vocab.txt "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
curl -L -o special_tokens_map.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json"
curl -L -o modules.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json"
curl -L -o config_sentence_transformers.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json"
curl -L -o sentence_bert_config.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json"

# Download pooling config
curl -L -o 1_Pooling/config.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json"

# Check what was downloaded
ls -lh
ls -lh 1_Pooling/


ls -la ~/Desktop/Testing_local_gemma_workings/all-MiniLM-L6-v2/
ls -la ~/Desktop/Testing_local_gemma_workings/all-MiniLM-L6-v2/1_Pooling/




cd ~/Desktop/Testing_local_gemma_workings/all-MiniLM-L6-v2

# Download safetensors version (safer and newer format)
curl -L -o model.safetensors "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"

# Remove the old pytorch_model.bin if it exists
rm -f pytorch_model.bin
