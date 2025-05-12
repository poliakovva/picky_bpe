use super::{super::OrderedVocabIter, trainer::PbpeTrainer, Error, Pair, Word};
use crate::pre_tokenizers::split;
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::{Cache, DEFAULT_CACHE_CAPACITY};
use regex::Split;
use serde_json::Value;
use std::borrow::Cow;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

pub type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;
pub type MergeMap = HashMap<Pair, Vec<(u32, u32)>>;
pub type SplitMap = HashMap<u32, Vec<(u32, Vec<u32>)>>;

struct Config {
    vocab: Vocab,
    merges: MergeMap,
    splits: SplitMap,
    cache_capacity: usize,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

/// A `PbpeBuilder` can be used to create a `PBPE` model with a custom configuration.
pub struct PbpeBuilder {
    config: Config,
}

impl Default for PbpeBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                vocab: HashMap::new(),
                merges: HashMap::new(),
                splits: HashMap::new(),
                cache_capacity: DEFAULT_CACHE_CAPACITY,
                unk_token: None,
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                ignore_merges: false,
            },
        }
    }
}

impl PbpeBuilder {
    /// Constructs a new `PbpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the vocab (token -> ID) and merges mappings.
    #[must_use]
    pub fn vocab_and_merges_and_splits(
        mut self,
        vocab: Vocab,
        merges: MergeMap,
        splits: SplitMap,
    ) -> Self {
        self.config.vocab = vocab;
        self.config.merges = merges;
        self.config.splits = splits;
        self
    }

    /// Set the cache's capacity. Set to 0 if you want to disable caching.
    #[must_use]
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.config.cache_capacity = capacity;
        self
    }

    /// Set the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// Set the `continuing_subword_prefix` option.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the `end_of_word_suffix` option.
    #[must_use]
    pub fn end_of_word_suffix(mut self, prefix: String) -> Self {
        self.config.end_of_word_suffix = Some(prefix);
        self
    }

    /// Set the `fuse_unk` option.
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.config.fuse_unk = fuse_unk;
        self
    }

    /// Set the `byte_fallback` option.
    #[must_use]
    pub fn byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.config.byte_fallback = byte_fallback;
        self
    }
    /// Set the `ignore_merges` option.
    #[must_use]
    pub fn ignore_merges(mut self, ignore_merges: bool) -> Self {
        self.config.ignore_merges = ignore_merges;
        self
    }

    /// Returns a `PBPE` model that uses the `PbpeBuilder`'s configuration.
    pub fn build(mut self) -> Result<PBPE> {
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        let cache = match self.config.cache_capacity {
            0 => None,
            capacity => Some(Cache::new(capacity)),
        };

        let vocab = self.config.vocab;

        Ok(PBPE {
            vocab,
            vocab_r,
            merges: self.config.merges,
            splits: self.config.splits,
            cache,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            fuse_unk: self.config.fuse_unk,
            byte_fallback: self.config.byte_fallback,
            ignore_merges: self.config.ignore_merges,
        })
    }
}

#[derive(PartialEq)]
pub struct PBPE {
    /// The vocabulary assigns a number to each token.
    pub(crate) vocab: Vocab,
    /// Reversed vocabulary, to rebuild sentences.
    pub(crate) vocab_r: VocabR,
    /// Contains the mapping between Pairs and their (rank, new_id).
    pub(crate) merges: MergeMap,
    /// Contains the mapping between token and their (rank, split_vector).
    pub(crate) splits: SplitMap,
    /// Contains the cache for optimizing the encoding step.
    pub(crate) cache: Option<Cache<String, Word>>,
    /// The unknown token to be used when we encounter an unknown char
    pub unk_token: Option<String>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to caracterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// Do multiple unk tokens get fused
    pub fuse_unk: bool,
    /// Byte fallback from sentence pieces, instead of UNK, uses `"<0x00>"`
    /// for each byte in the unk token
    pub byte_fallback: bool,
    /// Whether or not to direct output words if they are part of the vocab.
    pub ignore_merges: bool,
}

impl std::fmt::Debug for PBPE {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("PBPE")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("end_of_word_suffix", &self.end_of_word_suffix)
            .field("fuse_unk", &self.fuse_unk)
            .field("byte_fallback", &self.byte_fallback)
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .field("ignore_merges", &self.ignore_merges)
            .finish()
    }
}

impl Default for PBPE {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Clone for PBPE {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new PBPE will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.as_ref().map(|cache| cache.fresh());
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            splits: self.splits.clone(),
            cache: fresh_cache,
            unk_token: self.unk_token.clone(),
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        }
    }
}

impl PBPE {
    /// Initialize a `PbpeBuilder`.
    pub fn builder() -> PbpeBuilder {
        PbpeBuilder::new()
    }

    /// Create a new PBPE model with the given vocab and merges.
    pub fn new(vocab: Vocab, merges: MergeMap, splits: SplitMap) -> Self {
        Self::builder()
            .vocab_and_merges_and_splits(vocab, merges, splits)
            .build()
            .unwrap()
    }

    /// Reset the cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear()
        }
    }

    pub fn get_vocab(&self) -> Vocab {
        self.vocab.clone()
    }

    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    pub fn get_continuing_subword_prefix(&self) -> &Option<String> {
        &self.continuing_subword_prefix
    }

    fn merge_word(&self, w: &str) -> Result<Word> {
        let mut indices = w.char_indices().map(|(idx, _)| idx).peekable();
        let mut word = Word::with_capacity(w.len());
        let mut unk: Option<(u32, usize)> = None;
        while let Some(i) = indices.next() {
            let end = indices.peek();
            let is_first = i == 0;
            let is_last = end.is_none();

            let mut s = if let Some(e) = end {
                Cow::Borrowed(&w[i..*e]) //почему не просто w[i]??
            } else {
                Cow::Borrowed(&w[i..])
            };
            let byte_len = s.len();

            // Add the `continuing_subword_prefix` if relevant
            if !is_first {
                if let Some(ref prefix) = self.continuing_subword_prefix {
                    s = format!("{prefix}{s}").into()
                }
            }
            // Add the `end_of_word_suffix` if relevant
            if is_last {
                if let Some(ref suffix) = self.end_of_word_suffix {
                    s = format!("{s}{suffix}").into()
                }
            }

            if let Some(id) = self.vocab.get(s.as_ref()) {
                if let Some((unk_id, unk_len)) = unk {
                    word.add(unk_id, unk_len);
                    unk = None;
                }
                word.add(*id, byte_len);
            } else {
                if self.byte_fallback {
                    let tokens: Option<Vec<_>> = s
                        .bytes()
                        .map(|b| -> Option<&u32> {
                            let code = format!("<{b:#04X}>");

                            self.vocab.get(&code)
                        })
                        .collect();
                    if let Some(tokens) = tokens {
                        for t in tokens {
                            word.add(*t, 1);
                        }
                        continue;
                    }
                }
                if let Some(unk_token) = &self.unk_token {
                    unk = match (unk, self.fuse_unk) {
                        (Some((unk_id, unk_len)), true) => {
                            // Fuse unk
                            Some((unk_id, unk_len + byte_len))
                        }
                        (Some((unk_id, unk_len)), false) => {
                            // Do not fuse unk, add the previous one
                            word.add(unk_id, unk_len);
                            Some((
                                *self.vocab.get(unk_token).ok_or_else(|| {
                                    Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                                })?,
                                byte_len,
                            ))
                        }
                        _ => Some((
                            *self.vocab.get(unk_token).ok_or_else(|| {
                                Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                            })?,
                            byte_len,
                        )),
                    };
                }
            }
        }
        if let Some((unk_id, unk_len)) = unk {
            word.add(unk_id, unk_len);
        }

        word.merge_split_all(&self.merges, &self.splits);

        Ok(word)
    }

    fn word_to_tokens<'a, 'b: 'a>(&'a self, word: &'b Word) -> impl Iterator<Item = Token> + 'a {
        word.get_chars_iter()
            .zip(word.get_offsets_iter())
            .map(move |(id, offsets)| Token::new(id, self.vocab_r[&id].clone(), offsets))
    }

    fn tokenize_with_cache(&self, sequence: &str) -> Result<Vec<Token>> {
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(sequence) {
                return Ok(vec![Token::new(
                    *id,
                    sequence.to_string().clone(),
                    (0, sequence.len()),
                )]);
            }
        }
        if let Some(ref hit) = self.cache.as_ref().and_then(|c| c.get(sequence)) {
            return Ok(self.word_to_tokens(hit).collect());
        }
        let word = self.merge_word(sequence)?;
        let ret = self.word_to_tokens(&word).collect();
        if let Some(ref cache) = self.cache {
            cache.set(sequence.to_owned(), word);
        }
        Ok(ret)
    }
}

impl Model for PBPE {
    type Trainer = PbpeTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        self.tokenize_with_cache(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_trainer(&self) -> PbpeTrainer {
        PbpeTrainer::default()
    }

    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ordered_vocab_iter() {
        let vocab_r: VocabR = [
            (0, "a".into()),
            (1, "b".into()),
            (2, "c".into()),
            (3, "ab".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
        assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
    }

    #[test]
    fn test_unk_not_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, HashMap::new(), HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = pbpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = pbpe.tokenize("cc").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "<unk>".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
            ]
        );

        let tokens = pbpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
                Token::new(0u32, "<unk>".into(), (2, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }
    #[test]
    fn test_unk_get_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, HashMap::new(), HashMap::new())
            .unk_token("<unk>".to_string())
            .fuse_unk(true)
            .build()
            .unwrap();
        let tokens = pbpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = pbpe.tokenize("cc").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 2)),]);

        let tokens = pbpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }

    #[test]
    fn test_tokenize() {
        let vocab: Vocab = [
            ("u".into(), 0),
            ("n".into(), 1),
            ("r".into(), 2),
            ("e".into(), 3),
            ("l".into(), 4),
            ("a".into(), 5),
            ("t".into(), 6),
            ("d".into(), 7),
            ("re".into(), 8),
            ("at".into(), 9),
            ("ed".into(), 10),
            ("un".into(), 11),
            ("ated".into(), 12),
            ("rel".into(), 13),
            ("related".into(), 14),
            ("unrelated".into(), 15),
        ]
        .iter()
        .cloned()
        .collect();
        let merges: MergeMap = vec![
            // ("r", "e") -> "re", id 8
            ((2, 3), vec![(0, 8)]),
            // ("a", "t") -> "at", id 9
            ((5, 6), vec![(1, 9)]),
            // ("e", "d") -> "ed", id 10
            ((3, 7), vec![(2, 10)]),
            // ("u", "n") -> "un", id 11
            ((0, 1), vec![(3, 11)]),
            // ("at", "ed") -> "ated", id 12
            ((9, 10), vec![(4, 12)]),
            // ("re", "l") -> "rel", id 13
            ((8, 4), vec![(5, 13)]),
            // ("rel", "ated") -> "related", id 14
            ((13, 12), vec![(6, 14)]),
            // ("un", "related") -> "unrelated", id 15
            ((11, 14), vec![(7, 15)]),
        ]
        .into_iter()
        .collect();
        
        // No splits in this test
        let splits: SplitMap = HashMap::new();
        
        let mut pbpe = PBPE::new(vocab, merges, splits);

        // With no dropout:
        let tokens = pbpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);
    }

    #[test]
    fn test_pbpe_with_continuing_subword_prefix() {
        let vocab: Vocab = vec![
            ("a".to_string(), 0),
            ("##b".to_string(), 1),
            ("##c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges: MergeMap = vec![
            // ("a", "##b") -> "ab", id 3
            ((0, 1), vec![(0, 3)]),
            // ("ab", "##c") -> "abc", id 4
            ((3, 2), vec![(1, 4)]),
        ]
        .into_iter()
        .collect();

        let pbpe = PBPE::builder()
            .vocab_and_merges_and_splits(vocab, merges, HashMap::new())
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();

        let res = pbpe.tokenize("ab");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 3,
                value: "ab".to_string(),
                offsets: (0, 2)
            }]
        );
        let res = pbpe.tokenize("abc");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 4,
                value: "abc".to_string(),
                offsets: (0, 3)
            }]
        );
    }

    #[test]
    fn test_pbpe_byte_fallback() {
        // 0x61 == 'a' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x61>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, HashMap::new(), HashMap::new())
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = pbpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = pbpe.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x61>".into(), (0, 1)),]);
    }

    #[test]
    fn test_pbpe_byte_fallback_newline() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x0A>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, HashMap::new(), HashMap::new())
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = pbpe.tokenize("\n").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x0A>".into(), (0, 1)),]);
    }

    #[test]
    fn test_ignore_merges() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [
            (".:.:".into(), 0),
            ("Ġbelirtilen".into(), 1),
            (".".into(), 2),
            (":".into(), 3),
            ("bel".into(), 4),
            ("irtilen".into(), 5),
            ("Ġ".into(), 6),
            (".:".into(), 7),
            ("belirtilen".into(), 8),
            (".:.".into(), 9),
            ("be".into(), 10),
            ("l".into(), 11),
            ("ir".into(), 12),
            ("ti".into(), 13),
            ("en".into(), 14),
            ("irtil".into(), 15),
            ("irti".into(), 16),
            ("i".into(), 17),
            ("r".into(), 18),
            ("t".into(), 19),
            ("b".into(), 20),
            ("e".into(), 21),
            ("n".into(), 22),
        ]
        .iter()
        .cloned()
        .collect();
        let merges: MergeMap = vec![
            // (".", ":") -> ".:" (id 7)
            ((2, 3), vec![(0, 7)]),
            // ("b", "e") -> "be", id 10
            ((20, 21), vec![(0, 10)]),
            // ("be", "l") -> "bel", id 4
            ((10, 11), vec![(1, 4)]),
            // ("i", "r") -> "ir", id 12
            ((17, 18), vec![(2, 12)]),
            // ("t", "i") -> "ti", id 13
            ((19, 17), vec![(3, 13)]),
            // ("ir", "ti") -> "irti", id 16
            ((12, 13), vec![(4, 16)]),
            // ("irti", "l") -> "irtil", id 15
            ((16, 11), vec![(5, 15)]),
            // ("e", "n") -> "en", id 14
            ((21, 22), vec![(6, 14)]),
        ].into_iter().collect();
        let mut pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(
                vocab,
                merges,
                HashMap::new(),
            )
            .ignore_merges(true)
            .build()
            .unwrap();
        let tokens = pbpe.tokenize(".:.:").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, ".:.:".into(), (0, 4))]);

        let tokens = pbpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![Token::new(1u32, "Ġbelirtilen".into(), (0, 12))]
        );

        pbpe.ignore_merges = false;

        let tokens = pbpe.tokenize(".:.:").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(7u32, ".:".into(), (0, 2)),
                Token::new(7u32, ".:".into(), (2, 4))
            ]
        );

        let tokens = pbpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 6,
                    value: "Ġ".into(),
                    offsets: (0, 2)
                },
                Token {
                    id: 4,
                    value: "bel".into(),
                    offsets: (2, 5)
                },
                Token {
                    id: 15,
                    value: "irtil".into(),
                    offsets: (5, 10)
                },
                Token {
                    id: 14,
                    value: "en".into(),
                    offsets: (10, 12)
                }
            ]
        )
    }
}
