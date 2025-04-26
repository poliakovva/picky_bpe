# Picky BPE Tokenizer

<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>

A Rust implementation of the Picky BPE (Byte-Pair Encoding) tokenizer, an enhanced version of BPE

## What is Picky BPE?

Picky BPE is an enhanced version of the standard BPE tokenizer that adds several key features:

1. **Split Operations**: In addition to merges, PBPE supports splitting tokens back into their components, allowing for more flexible tokenization strategies.
2. **Continuing Subword Prefix**: Support for prefix markers (like "##" in BERT) to distinguish between word-initial and word-continuing subwords.
3. **End of Word Suffix**: Optional suffix markers to distinguish end-of-word subwords.
4. **Token Length Control**: Ability to limit the maximum length of generated tokens.
5. **Byte Fallback**: Fallback to byte-level tokenization for unknown characters.
6. **UNK Token Fusion**: Option to fuse multiple unknown tokens into a single token.

## Key Features

- **Flexible Tokenization**: Support for both merging and splitting operations
- **Configurable Prefixes/Suffixes**: Customizable markers for subword parts
- **Length Control**: Enforce maximum token lengths
- **Parallel Processing**: Leverages CPU parallelism for faster tokenization
- **Memory Efficient**: Uses caching to optimize repeated tokenizations

## Usage Examples



### Training a New Model

```rust
use tokenizers::models::pbpe::{PbpeTrainer, PBPE};

fn main() -> Result<()> {
    let trainer = PbpeTrainer::builder()
        .vocab_size(30000)
        .min_frequency(0)
        .max_token_length(Some(16))
        .show_progress(true)
        .build();

    let mut model = PBPE::default();
    trainer.do_train(&word_counts, &mut model)?;
    Ok(())
}
```

### Advanced Configuration

```rust
use tokenizers::models::pbpe::{PBPE, PbpeBuilder};

fn main() -> Result<()> {
    let pbpe = PbpeBuilder::default()
        .vocab_and_merges_and_splits(vocab, merges, splits)
        .continuing_subword_prefix("##".to_string())
        .end_of_word_suffix("</w>".to_string())
        .unk_token("[UNK]".to_string())
        .fuse_unk(true)
        .byte_fallback(true)
        .ignore_merges(false)
        .build()?;
    Ok(())
}
```

## Performance

The implementation is optimized for performance:
- Uses parallel processing for training and tokenization
- Implements efficient caching for repeated tokenizations
- Minimizes memory allocations during processing

## Configuration Options

- `vocab_size`: Target vocabulary size
- `min_frequency`: Minimum frequency for tokens
- `max_token_length`: Maximum length for generated tokens
- `continuing_subword_prefix`: Prefix for continuing subwords
- `end_of_word_suffix`: Suffix for end-of-word tokens
- `unk_token`: Token for unknown characters
- `fuse_unk`: Whether to fuse multiple unknown tokens
- `byte_fallback`: Whether to use byte-level fallback
- `ignore_merges`: Whether to ignore merge operations

## Additional Information

- The implementation leverages CPU parallelism when possible
- Parallelism can be tuned using the `RAYON_RS_NUM_THREADS` environment variable
- Progress bar visualization is enabled by default (can be disabled)

## Features

- **progressbar**: Enabled by default for training visualization
- **parallel**: Parallel processing support for improved performance
- **cache**: Tokenization caching for repeated operations
