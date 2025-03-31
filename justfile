#!/usr/bin/env just --justfile
set quiet
# set dotenv-required
# set dotenv-load := true
# set dotenv-filename := ".env"


default:
    just --list

[doc('Content-based filtering')]
run-content DATA_DIR="data/sample_data/":
    echo "Data directory: {{DATA_DIR}}"
    sleep 1
    echo "Running content based filtering recommendation"
    sleep 1
    echo "Here are the recommendations"

[doc('Collaborative filtering')]
run-collab DATA_DIR="data/sample_data/":
    echo "Data directory: {{DATA_DIR}}"
    sleep 1
    echo "Running collaborative filtering recommendation"
    sleep 1
    echo "Here are the recommendations"


[doc('Deep neural network')]
run-dnn DATA_DIR="data/sample_data/":
    echo "Data directory: {{DATA_DIR}}"
    sleep 1
    echo "Running deep neural network recommendation"
    sleep 1
    echo "Here are the recommendations"

[doc('Retrieval and ranking')]
run-rr DATA_DIR="data/sample_data/":
    echo "Data directory: {{DATA_DIR}}"
    sleep 1
    echo "Running retrieval and rankign recommendation"
    sleep 1
    echo "Here are the recommendations"
    
[doc('TIGER recommendation')]
run-tiger:
    echo "Running tiger recommendation"
    sleep 1
    echo "Here are the recommendations"

[doc('Linting')]
lint:
    flake8 .
