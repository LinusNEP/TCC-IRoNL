#!/bin/bash

TCC_Path=$(rospack find tcc-ironl)

# Path to the scripts
clip_node="python3 $TCC_Path/vlm_nodes/clip_node.py"
glip_node="python3 $TCC_Path/vlm_nodes/glip_node.py"

show_menu() {
    echo "Select the VLM to run:"
    echo "0) Exit"
    echo "1) CLIP"
    echo "2) GLIP"
}

run_selected_script() {
    local choice
    read -p "Enter choice [0-2]: " choice
    case $choice in
    	0) exit 0 ;;
        1) $clip_node ;;
        2) $glip_node ;;
        *) echo "Invalid choice. Please select a number between 0-2." ;;
    esac
}

while true
do
    show_menu
    run_selected_script
done

