#!/bin/bash

TCC_Path=$(rospack find tcc-ironl)

# Path to the scripts
gpt2_llm_node="python3 $TCC_Path/llm_nodes/scripts/gpt2_llm.py"
bert_llm_node="python3 $TCC_Path/llm_nodes/bert_llm.py"
distilBERT_llm_node="python3 $TCC_Path/llm_nodes/distilBERT_llm.py"
gptNeo_llm_node="python3 $TCC_Path/llm_nodes/gptNeo_llm.py"
llama_llm_node="python3 $TCC_Path/llm_nodes/llama_llm.py"
RoBERTa_llm_node="python3 $TCC_Path/llm_nodes/RoBERTa_llm.py"
gpt3_llm_node="python3 $TCC_Path/llm_nodes/gpt3_llm.py"

# Select from the menu
show_menu() {
    echo "Select the LLM to run:"
    echo "0) Exit"
    echo "1) OpenAI GPT-2"
    echo "2) Google BERT"
    echo "3) HuggingFace distilBERT"
    echo "4) EleutherAI GPTNeo"
    echo "5) Meta AI LLaMA"
    echo "6) Facebook RoBERTa"
    echo "7) OpenAI GPT-3"
}

# User choice
run_selected_script() {
    local choice
    read -p "Enter choice [0-7]: " choice
    case $choice in
    	0) exit 0 ;;
        1) $gpt2_llm_node ;;
        2) $bert_llm_node ;;
        3) $distilBERT_llm_node ;;
        4) $gptNeo_llm_node ;;
        5) $llama_llm_node ;;
        6) $RoBERTa_llm_node ;;
        7) $gpt3_llm_node ;;
        *) echo "Invalid choice. Please select a number between 0-7." ;;
    esac
}

# Main loop
while true
do
    show_menu
    run_selected_script
done

