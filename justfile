#!/usr/bin/env just --justfile
set quiet
set dotenv-required
set dotenv-filename := ".env"

default:
    just --list

[doc('Content-based filtering')]
run-content:
    python src/content_based.py

[doc('Collaborative filtering')]
run-collab:
    python src/collaborative.py

[doc('Deep neural network')]
run-dnn:
    python src/dnn.py

[doc('Retrieval and ranking')]
run-rr:
    python src/retrieval_rank.py

[doc('Linting')]
lint:
    flake8 src/

