#![allow(warnings)]
use std::fs;
use std::io;
use std::path::Path;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
use tokenizers::parallelism;
use tokenizers::parallelism::get_parallelism;
use tokenizers::parallelism::set_parallelism;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::TokenizerImpl;
use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};
use clap::Parser;

use std::io::{Read};

fn is_valid_utf8(file_path: &Path) -> io::Result<bool> {
    let mut file = fs::File::open(file_path)?;
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    Ok(String::from_utf8(contents).is_ok())
}

fn filter_utf8_txt_files(dir_path: &str) -> io::Result<Vec<String>> {
    let mut valid_files = Vec::new();

    // Read the directory contents
    for entry in fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        // Check if the path is a file and has a .txt extension
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            // Check if the file is valid UTF-8
            if is_valid_utf8(&path)? {
                valid_files.push(path.display().to_string());
            }
        }
    }

    Ok(valid_files)
}

#[derive(Parser)]
struct Cli {
    dir_path: String,
    vocab_size: usize,
}


fn main() -> Result<()> {
    use std::time::Instant;
    let now = Instant::now();
    let args = Cli::parse();

    let dir_path = args.dir_path;
    let vocab_size: usize = args.vocab_size;

    let files = filter_utf8_txt_files(&dir_path).unwrap();

    let mut trainer = BpeTrainer::builder()
    .vocab_size(vocab_size)
    .special_tokens(vec![
        AddedToken::from("[UNK]", true),
        AddedToken::from("[BOS]", true),
        AddedToken::from("[EOS]", true),
        AddedToken::from("[PAD]", true),
    ])
    .show_progress(true)
    .build();


    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit {}));

    // let mut tokenizer = TokenizerBuilder::new()
    // .with_model(BPE::default())
    // .with_normalizer(Some(Sequence::new(vec![
    //     Strip::new(true, true).into(),
    //     NFC.into(),
    // ])))
    // .with_pre_tokenizer(Some(ByteLevel::default()))
    // .with_post_processor(Some(ByteLevel::default()))
    // .with_decoder(Some(ByteLevel::default()))
    // .build()?;

    let pretty = false;
    set_parallelism(true);
    tokenizer
        .train_from_files(&mut trainer, files)?;
    let elapsed = now.elapsed();
    println!("Total time: {:.2?}", elapsed);
    Ok(())
}