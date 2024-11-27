#!/bin/bash

# Run the first command

cargo run --manifest-path ./playground_picky/Cargo.toml --release -- "./datasets/D184MB" 1000 >> ./logs/picky_logs.txt
echo "Finished running PickyBPE on 184MB dataset with vocab_size=1000" >> ./logs/picky_logs.txt

# Run the second command
cargo run --manifest-path ./playground/Cargo.toml --release -- "./datasets/D184MB" 1000 >> ./logs/HF_logs.txt
echo "Finished running HF_BPE on 184MB dataset with vocab_size=1000" >> ./logs/HF_logs.txt

# Run the third command
python3 ./picky_bpe/bpe_trainer.py --input_file ./picky_bpe/combined_184.txt --model_file model.json --vocab_size 1000 --threshold 0.7 --coverage 1 >> ./logs/python_logs.txt
echo "Finished running Python PickyBPE on 184MB dataset with vocab_size=1000" >> ./logs/python_logs.txt


# Run the 4 command

cargo run --manifest-path ./playground_picky/Cargo.toml --release -- "./datasets/D670MB" 10000 >> ./logs/picky_logs.txt
echo "Finished running PickyBPE on 670MB dataset with vocab_size=10000" >> ./logs/picky_logs.txt

# Run the 5 command
cargo run --manifest-path ./playground/Cargo.toml --release -- "./datasets/D670MB" 10000 >> ./logs/HF_logs.txt
echo "Finished running HF_BPE on 670MB dataset with vocab_size=10000" >> ./logs/HF_logs.txt

# Run the 6 command
python3 ./picky_bpe/bpe_trainer.py --input_file ./picky_bpe/combined_670.txt --model_file model.json --vocab_size 10000 --threshold 0.7 --coverage 1 >> ./logs/python_logs.txt
echo "Finished running Python PickyBPE on 670MB dataset with vocab_size=10000" >> ./logs/python_logs.txt

# Run the 7 command

cargo run --manifest-path ./playground_picky/Cargo.toml --release -- "./datasets/D670MB" 30000 >> ./logs/picky_logs.txt
echo "Finished running PickyBPE on 670MB dataset with vocab_size=30000" >> ./logs/picky_logs.txt

# Run the 8 command
cargo run --manifest-path ./playground/Cargo.toml --release -- "./datasets/D670MB" 30000 >> ./logs/HF_logs.txt
echo "Finished running HF_BPE on 670MB dataset with vocab_size=30000" >> ./logs/HF_logs.txt
