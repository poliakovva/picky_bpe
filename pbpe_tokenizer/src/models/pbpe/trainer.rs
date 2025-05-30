#![allow(clippy::map_entry)]

use super::{Pair, WithFirstLastIterator, Word, PBPE};
use crate::parallelism::*;
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::progress::{ProgressBar, ProgressStyle};
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::collections::{HashMap, HashSet};
use std::process::id;
use std::sync::atomic;

enum EventType {
    Merge = 0,
    Split = 1,
}

#[derive(Debug)]
struct Merge {
    pair: Pair,
    pos: Vec<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.pair == other.pair
    }
}
impl Eq for Merge {}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        other.pair.cmp(&self.pair)
    }
}
impl Hash for Merge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pair.hash(state);
    }
}

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
    tau: Option<f64>,
}

/// A `PbpeTrainerBuilder` can be used to create a `PbpeTrainer` with a custom
/// configuration.
pub struct PbpeTrainerBuilder {
    config: Config,
}

impl Default for PbpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                tau: Some(1f64),
            },
        }
    }
}

impl PbpeTrainerBuilder {
    /// Constructs a new `PbpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }
    /// Set tau
    #[must_use]
    pub fn tau(mut self, tau: Option<f64>) -> Self {
        self.config.tau = tau;
        self
    }

    /// Constructs the final PbpeTrainer
    pub fn build(self) -> PbpeTrainer {
        PbpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            tau: self.config.tau,
            words: HashMap::new(),
        }
    }
}

/// In charge of training a `PBPE` model
///
/// # Examples
///
/// ```
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::pbpe::{PBPE, PbpeTrainer};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = PbpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// let mut model = PBPE::default();
/// let special_tokens = trainer.train(&mut model).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PbpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// A list of special tokens that the model should know of
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: HashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to caracterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,
    /// Threshold for IoS metric
    pub tau: Option<f64>,

    words: HashMap<String, u64>,
}

impl Default for PbpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl PbpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> PbpeTrainerBuilder {
        PbpeTrainerBuilder::new()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, w2id: &mut HashMap<String, u32>, id2w: &mut Vec<String>) {
        for token in &self.special_tokens {
            if !w2id.contains_key(&token.content) {
                id2w.push(token.content.to_owned());
                w2id.insert(token.content.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            alphabet
                .entry(*c)
                .and_modify(|cnt| *cnt = usize::MAX)
                .or_insert(usize::MAX);
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if alphabet.len() > limit {
                    alphabet.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
        });
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &HashMap<String, u64>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&s) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s = format!("{prefix}{s}");
                        }
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s = format!("{s}{suffix}");
                        }
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&s) {
                        id2w.push(s.clone());
                        w2id.insert(s.clone(), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&s], 1); // We do not care about the len here
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (HashMap<Pair, i32>, HashMap<Pair, HashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = HashMap::new();
                let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    if !pair_counts.contains_key(&cur_pair) {
                        pair_counts.insert(cur_pair, 0);
                    }

                    // Then update counts
                    let count = counts[i];
                    where_to_update
                        .entry(cur_pair)
                        .and_modify(|h| {
                            h.insert(i);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(i);
                            h
                        });
                    *pair_counts.get_mut(&cur_pair).unwrap() += count as i32;
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (HashMap::new(), HashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        pair_counts.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    for (k, v) in wtu {
                        where_to_update
                            .entry(k)
                            .and_modify(|set| *set = set.union(&v).copied().collect())
                            .or_insert(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub fn split(
        token_id: u32,
        parents: &Vec<Pair>,
        id_active: &Vec<u8>,
        atomic_size: u32,
    ) -> Vec<usize> {
        if (token_id < atomic_size) || (id_active[(token_id-atomic_size) as usize] == 1) {
            vec![token_id as usize]
        } else {
            let (token1, token2) = parents[(token_id - atomic_size) as usize];
            [
                Self::split(token1, &parents, &id_active, atomic_size),
                Self::split(token2, &parents, &id_active, atomic_size),
            ]
            .concat()
        }
    }

    pub fn do_train(
        &self,
        word_counts: &HashMap<String, u64>,
        model: &mut PBPE,
    ) -> Result<Vec<AddedToken>> {
        // println!("Tau: {}", self.tau.unwrap());
        // println!("vocab_size: {}", self.vocab_size);
        let mut word_to_id: HashMap<String, u32> = HashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<String> = Vec::with_capacity(self.vocab_size);
        //vec that has 1 is token is active or 0 if not
        let mut id_active: Vec<u8> = Vec::new();
        //vec of parents
        let mut parents: Vec<Pair> = Vec::with_capacity(self.vocab_size);

        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);
        let mut actual_vocab_size: usize;

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);
        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len());

        actual_vocab_size = id_to_word.len();
        let atomic_size = actual_vocab_size as u32;
        println!(
            "Initialized vocabulary with: {} unique characters",
            atomic_size
        );
        println!("Number of unique words:{}", word_counts.len());
        let mut sorted_words: Vec<(String, u32)> = word_to_id.clone().into_iter().collect();
        sorted_words.sort_by(|a, b| a.0.cmp(&b.0));

        // Print the sorted results
        for (word, id) in sorted_words {
            println!("Word: |{:?}| ID: {}", word, id);
        }

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // Insert them in the queue
        let mut queue = PriorityQueue::with_capacity(pair_counts.len());

        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(
                    Merge {
                        pair,
                        pos: pos.into_iter().collect::<Vec<_>>(),
                    },
                    count,
                );
            }
        });

        self.finalize_progress(&progress, words.len());

        //
        // 5. Do merges and splits
        //
        let mut count_merged_tokens: HashMap<u32, i32> = HashMap::with_capacity(pair_counts.len()); /////// -------------
        self.update_progress(&progress, self.vocab_size, "Compute events");
        let mut merges: Vec<(Pair, u32)> = vec![];
        let mut events: Vec<(EventType, usize)> = vec![];
        let mut splits: Vec<(u32, Vec<usize>)> = vec![];
        let mut removes_counter = 0;

        loop {
            // Stop as soon as we have a big enough vocabulary
            if actual_vocab_size >= self.vocab_size {
                break;
            }

            if queue.is_empty() {
                break;
            }
            let top = queue.pop().unwrap();
            let mut priority = top.1;
            let top = top.0;
            // println!("priority: {}", priority);
            // println!("pair_counts: {}", pair_counts[&top.pair]);
            // if priority != pair_counts[&top.pair] {
            //     priority = pair_counts[&top.pair];
            //     if priority > 0 {
            //         queue.push(top, priority);// expect
            //     }
            //     continue;
            // }

            if priority < 1
                || self.min_frequency > priority as u64
                || (top.pair.0 >= atomic_size
                    && id_active[(top.pair.0 - atomic_size) as usize] == 0)
                || (top.pair.1 >= atomic_size
                    && id_active[(top.pair.1 - atomic_size) as usize] == 0)
            {
                continue;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].to_owned();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{part_a}{part_b}");
            // implement sentencepiece-like merge.
            // if this code were to be merged, integrate a way in the python bindings to communicate this variable
            // default should be 0/None to maintain previous behavior. 16 is the spm default.

            // Insert new token if it does not already exist
            let new_token_id = word_to_id
                .get(&new_token)
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&new_token) {
                id_to_word.push(new_token.clone());
                word_to_id.insert(new_token.clone(), new_token_id);
                parents.push(top.pair);
                id_active.push(1);
                println!(
                    "merged tokens: '{}' with token '{}'| count: {}",
                    id_to_word[top.pair.0 as usize], id_to_word[top.pair.1 as usize], priority
                );
            } else {
                id_active[(new_token_id - atomic_size) as usize] = 1;
                println!(
                    "restored token: '{}'| count: {}",
                    new_token, priority
                );
            }
            actual_vocab_size += 1;

            merges.push((top.pair, new_token_id));
            events.push((EventType::Merge, merges.len() - 1));
            count_merged_tokens
                .entry(new_token_id)
                .and_modify(|c| *c += pair_counts[&top.pair]) //??
                .or_insert(pair_counts[&top.pair]);

            // Merge the new pair in every words
            // Safety: This is just a type assertion, the code below may no longer be safe
            // if the type of `pos` changes
            let pos = &top.pos;
            

            let words_len = words.len();
            struct WordPtr(*mut Word);
            // Safety: We do not actually use this for concurrent access to the same memory,
            // only to different chunks within the same allocation.
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            let mut changes = pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    // Safety:
                    // We are producing a valid pointer since we are indexing in bounds
                    //
                    // We can access each `word` here in parallel because each position
                    // can be there only once (pos is a HashSet).
                    unsafe {
                        assert!(i < words_len);
                        // This is words[i], but avoids needing to go through &T (which triggers UB)
                        let word = word_start.0.add(i);
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();
            //check IOS metric for token1 and token2.
            //then remove the x from vocab and add new event to merges
            if let Some(&count_1) = count_merged_tokens.get(&top.pair.0) {
                //update count on merge
                count_merged_tokens
                    .entry(top.pair.0)
                    .and_modify(|c| *c -= pair_counts[&top.pair]);
                // println!("TOKEN TO REMOVE: {:#?}, IoS: {}", id_to_word[top.pair.0 as usize], (priority as f64)/f64::from(count_1));

                if (priority as f64) / f64::from(count_1) >= self.tau.unwrap()
                    && (top.pair.0 >= atomic_size)
                {
                    println!(
                        "removed token: {:#?} with freq: {}, merged to {} with freq:{}",
                        id_to_word[top.pair.0 as usize],
                        f64::from(count_1),
                        new_token,
                        (priority as f64)
                    );

                    removes_counter += 1;
                    // println!("REMOVED: {:#?}", &top.pair.0);

                    id_active[(top.pair.0 - atomic_size) as usize] = 0;
                    //split token
                    let split_token = Self::split(top.pair.0, &parents, &id_active, atomic_size);
                    //add event
                    splits.push((top.pair.0, split_token.clone()));
                    events.push((EventType::Split, splits.len() - 1));

                    //clean the corpus x_1
                    let mut changes_token1 = words
                        .maybe_par_iter_mut()
                        .enumerate()
                        .map(|(i, word)| {
                            (*word)
                                .remove(top.pair.0, &split_token, max_token_length)
                                .into_iter()
                                .map(|c| (c, i))
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect::<Vec<_>>();
                    changes.append(&mut changes_token1);
                    actual_vocab_size -= 1;
                    //update count on remove
                    for token in split_token {
                        if let Some(value) = count_merged_tokens.get_mut(&(token as u32)) {
                            *value -= priority as i32;
                        }
                    }
                }
            }
            if top.pair.1 != top.pair.0 {
                if let Some(&count_2) = count_merged_tokens.get(&top.pair.1) {
                    // println!("SEEN: {:#?}", &top.pair.1);
                    // println!("TOKEN TO REMOVE: {:#?}, IoS: {}", id_to_word[top.pair.1 as usize], (priority as f64)/f64::from(count_2));
                    // println!("SELF.TAU: {:#?}", self.tau.unwrap());
                    // println!("atomic_size: {:#?}", atomic_size);

                    //update count on merge
                    count_merged_tokens
                        .entry(top.pair.1)
                        .and_modify(|c| *c -= pair_counts[&top.pair]);

                    if (priority as f64) / f64::from(count_2) >= self.tau.unwrap()
                        && (top.pair.1 >= atomic_size)
                    {
                        println!(
                            "removed token: {:#?} with freq: {}, merged to {} with freq:{}",
                            id_to_word[top.pair.1 as usize],
                            f64::from(count_2),
                            new_token,
                            (priority as f64)
                        );

                        removes_counter += 1;
                        id_active[(top.pair.1 - atomic_size) as usize] = 0;
                        //split token
                        let split_token =
                            Self::split(top.pair.1, &parents, &id_active, atomic_size);
                        //add event
                        splits.push((top.pair.1, split_token.clone()));
                        events.push((EventType::Split, splits.len() - 1));
                        //clean the corpus x_1
                        let mut changes_token2 = words
                            .maybe_par_iter_mut()
                            .enumerate()
                            .map(|(i, word)| {
                                (*word)
                                    .remove(top.pair.1, &split_token, max_token_length)
                                    .into_iter()
                                    .map(|c| (c, i))
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>();
                        changes.append(&mut changes_token2);
                        actual_vocab_size -= 1;
                        //update counts on remove
                        for token in split_token {
                            if let Some(value) = count_merged_tokens.get_mut(&(token as u32)) {
                                *value -= priority as i32;
                            }
                        }
                    }
                }
            }
            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;

                pair_counts
                    .entry(pair)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
                //modify token_counts
                //TODO!
                //check priority_queue +- priority
                if change > 0 {
                    where_to_update
                        .entry(pair)
                        .and_modify(|h| {
                            h.insert(iw);
                        })
                        .or_insert_with(|| {
                            let mut h = HashSet::new();
                            h.insert(iw);
                            h
                        });
                }
                // println!("pair {} {}, count {}", pair.0, pair.1, pair_counts[&pair]);

                // println!("found item in queue {:?}", queue.get_priority(&Merge{ pair, pos: where_to_update[&pair].iter().copied().collect::<Vec<_>>() }));

                 queue.change_priority(&Merge{ pair, pos: Vec::new()}, pair_counts[&pair]);

                
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count >= 0 {

                     queue.push(
                        Merge {
                            pair,
                            pos: pos.into_iter().collect::<Vec<_>>(),
                        },
                        count,);
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, events.len());

        println!("Removed {} tokens", removes_counter);

        let mut merges_map: HashMap<Pair, Vec<(u32, u32)>> = HashMap::new();
        let mut splits_map: HashMap<u32, Vec<(u32, Vec<u32>)>> = HashMap::new();

        for (event_id, (event_type, index)) in events.iter().enumerate() {
            match event_type {
                EventType::Merge => {
                    let (pair, new_id) = &merges[*index];
                    merges_map
                        .entry(*pair)
                        .or_insert_with(Vec::new)
                        .push((event_id as u32, *new_id));
                }
                EventType::Split => {
                    let (token_id, split_tokens) = &splits[*index];
                    splits_map
                        .entry(token_id.clone())
                        .or_insert_with(Vec::new)
                        .push((
                            event_id as u32,
                            split_tokens.iter().map(|&i| i as u32).collect(),
                        ));
                }
            }
        }

        // Transfer new vocab & options to mode
        model.vocab = word_to_id;

        model.vocab_r = model
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        model.merges = merges_map;
        model.splits = splits_map;

        if let Some(prefix) = &self.continuing_subword_prefix {
            model.continuing_subword_prefix = Some(prefix.to_owned());
        } else {
            model.continuing_subword_prefix = None;
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            model.end_of_word_suffix = Some(suffix.to_owned());
        } else {
            model.end_of_word_suffix = None;
        }

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for PbpeTrainer {
    type Model = PBPE;

    /// Train a PBPE model
    fn train(&self, model: &mut PBPE) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::models::pbpe::Word;

    use super::{Pair, PbpeTrainer, PBPE};
    use std::collections::HashMap;

    #[test]
    fn test_train() {
        let word_counts: HashMap<String, u64> = [
            ("roses".into(), 1),
            ("are".into(), 15),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = PbpeTrainer::builder()
            .show_progress(false)
            .tau(Some(0.3))
            .min_frequency(2)
            .build();
        let mut model = PBPE::default();
        trainer.do_train(&word_counts, &mut model).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: HashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),  
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.vocab, expected_vocab);

        // The keys in `merges` are pairs of symbols, the values are tuples of (rank, id),
        // where 'rank' determines the order in which this merge will be applied during
        // tokenization, and 'id' is the vocab id of the symbol resulting from merging
        // the pair of symbols in the corresponding key.
        let expected_merges: HashMap<Pair, Vec<(u32, u32)>> = [
            ((17, 11), vec![(0, 22)]), // 'r' + 'e'  -> 're'
            ((8, 22), vec![(1, 23)]),  // 'a' + 're' -> 'are'
            ((13, 18), vec![(3, 24)]), // 'i' + 's'  -> 'is'
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.merges, expected_merges);
        let expected_splits: HashMap<u32, Vec<(u32, Vec<u32>)>>  = [
            (22, vec![(2, vec![17, 11])])
        ].iter().cloned().collect();
        assert_eq!(model.splits, expected_splits);
    }

    #[test]
    fn test_merge_multiple_pairs() {
        // Let's use the word 'mississippi' and a word-to-id vocab:
        // {'m': 0, 'i': 1, 's': 2, 'p': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'm'
        word.add(1, 1); // 'i'
        word.add(2, 1); // 's'
        word.add(2, 1); // 's'
        word.add(1, 1); // 'i'
        word.add(2, 1); // 's'
        word.add(2, 1); // 's'
        word.add(1, 1); // 'i'
        word.add(3, 1); // 'p'
        word.add(3, 1); // 'p'
        word.add(1, 1); // 'i'

        // First merge the pair ('s', 's') ~= (2, 2), new ID is 4.
        let _ = word.merge(2, 2, 4, usize::MAX);

        // Then merge the pair ('p', 'p') ~= (3, 3), new ID is 5.
        let changes = word.merge(3, 3, 5, usize::MAX);

        // The word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'm'
                1u32, // 'i'
                4u32, // 'ss'
                1u32, // 'i'
                4u32, // 'ss'
                1u32, // 'i'
                5u32, // 'pp'
                1u32, // 'i'
            ]
        );

        // Check changes in pair counts after the second merge:
        assert_eq!(
            changes,
            &[
                ((3u32, 3u32), -1i32), // count for ('p', 'p') decreased.
                ((1u32, 3u32), -1i32), // count for ('i', 'p') decreased.
                ((1u32, 5u32), 1i32),  // count for ('i', 'pp') increased.
                ((3u32, 1u32), -1i32), // count for ('o', 'i') decreased.
                ((5u32, 1u32), 1i32),  // count for ('pp', 'i') increased.
            ]
        );
    }

    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: HashMap<String, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // korean 10 char
            ("긴한글문자열긴한글문자열긴한글문", 2), // korean 16 char
            ("短字符串短字符串短字", 2),             //simplified chinese 10 char
            ("长字符串长字符串长字符串长字符串", 2), // simp. chinese 16 char
            ("短い文字列短い文字列", 2),             // japanese 10 char
            ("長い文字列長い文字列長い文字列長", 2), // japanese 16 char
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect();
        let trainer = PbpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = PBPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let vocab = model.get_vocab();
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }
    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        let long_word_counts: HashMap<String, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (key.to_string(), *value))
        .collect();
        let trainer = PbpeTrainer::builder()
            .max_token_length(Some(2))
            .show_progress(false)
            .min_frequency(0)
            .build();
        let mut model = PBPE::default();
        trainer.do_train(&long_word_counts, &mut model).unwrap();
        let trained_vocab: HashMap<String, u32> = model.get_vocab();
        
        // Check that all basic characters are present
        let basic_chars = [
            "A", "G", "L", "P", "S", "i", "n", "o", "s",
            "い", "字", "文", "短", "符", "長", "长", "글", "긴", "은", "짧", "한"
        ];
        
        for char in basic_chars.iter() {
            assert!(trained_vocab.contains_key(*char), "Basic character {} is missing", char);
        }
        
        // Check that all tokens have length <= 2
        for token in trained_vocab.keys() {
            assert!(
                token.chars().count() <= 2,
                "Token {} is too long: {} chars",
                token,
                token.chars().count()
            );
        }
        
        // Check that some expected merges are present (without enforcing specific order)
        let expected_merges = [
            "so", "GP", "Lo", "in", "字符", "い文", "긴한"
        ];
        
        let mut found_merges = 0;
        for merge in expected_merges.iter() {
            if trained_vocab.contains_key(*merge) {
                found_merges += 1;
            }
        }
        
        // We should have at least some of the expected merges
        assert!(
            found_merges >= 3,
            "Expected at least 3 merges to be present, found {}",
            found_merges
        );
    }
    #[test]
    fn test_split() {
        let parents = vec![(2u32, 2u32), (0u32, 1u32), (1u32, 4u32)];
        let id_active = vec![0, 1, 0];
        let atomic_size = 4;
        let split_token = PbpeTrainer::split(6, &parents, &id_active, atomic_size);
        assert_eq!(split_token, vec![1, 2, 2]);
    }
}
