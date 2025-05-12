use super::Pair;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // By manually implementing this, we make the containing BinaryHeap a
        // min-heap ordered first on the rank, and the pos otherwise
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank != other.rank {
            other.rank.cmp(&self.rank)
        } else {
            other.pos.cmp(&self.pos)
        }
    }
}

#[derive(Debug, Eq)]
struct Split {
    pos: usize,
    rank: u32,
    split: Vec<u32>,
}

impl PartialEq for Split {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Split {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // By manually implementing this, we make the containing BinaryHeap a
        // min-heap ordered first on the rank, and the pos otherwise
        Some(self.cmp(other))
    }
}

impl Ord for Split {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank != other.rank {
            other.rank.cmp(&self.rank)
        } else {
            self.pos.cmp(&self.pos)
        }
    }
}

#[derive(Debug, Eq)]
enum Event {
    Merge(Merge),
    Split(Split),
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Event::Merge(a), Event::Merge(b)) => {
                a.rank.cmp(&b.rank).then_with(|| a.pos.cmp(&b.pos))
            }
            (Event::Split(a), Event::Split(b)) => {
                a.rank.cmp(&b.rank).then_with(|| a.pos.cmp(&b.pos))
            }
            (Event::Merge(a), Event::Split(b)) => a
                .rank
                .cmp(&b.rank)
                .then_with(|| a.pos.cmp(&b.pos))
                .then_with(|| Ordering::Less),
            (Event::Split(a), Event::Merge(b)) => a
                .rank
                .cmp(&b.rank)
                .then_with(|| a.pos.cmp(&b.pos))
                .then_with(|| Ordering::Greater),
        }
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}
impl Symbol {
    /// Merges the current Symbol with the other one.
    /// In order to update prev/next, we consider Self to be the Symbol on the left,
    /// and other to be the next one on the right.
    pub fn merge_with(&mut self, other: &Self, new_c: u32) {
        self.c = new_c;
        self.len += other.len;
        self.next = other.next;
    }
}

#[derive(Clone, Default)]
pub(super) struct Word {
    symbols: Vec<Symbol>,
}
impl std::fmt::Debug for Word {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Word")
            .field(
                "chars",
                &self
                    .symbols
                    .iter()
                    .map(|s| s.c.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
            .field("symbols", &self.symbols)
            .finish()
    }
}

impl Word {
    pub(super) fn new() -> Self {
        Word { symbols: vec![] }
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn add(&mut self, c: u32, byte_len: usize) {
        let (prev, next) = {
            let len = self.symbols.len() as isize;
            if let Some(last) = self.symbols.last_mut() {
                // Update `next` on the previous one
                last.next = len;
                (len - 1, -1)
            } else {
                (-1, -1)
            }
        };
        self.symbols.push(Symbol {
            c,
            prev,
            next,
            len: byte_len,
        });
    }

    pub(super) fn merge(
        &mut self,
        c1: u32,
        c2: u32,
        replacement: u32,
        max_length: usize,
    ) -> Vec<(Pair, i32)> {
        let mut changes: Vec<(Pair, i32)> = vec![];
        let mut i = 0;
        changes.push(((c1, c2), -1));//intentional
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Found a pair
            if self.symbols[i].c == c1 && i + 1 < self.symbols.len() && self.symbols[i + 1].c == c2
            {
                let first = self.symbols[i];
                let second = self.symbols[i + 1];

                // Remove in place
                let new_s = Symbol {
                    c: replacement,
                    prev: first.prev,
                    next: second.next,
                    len: first.len + second.len,
                };

                // If there are other characters before the pair
                if i > 0 {
                    changes.push(((self.symbols[i - 1].c, first.c), -1));
                    if self.symbols[i - 1].len + new_s.len < max_length {
                        changes.push(((self.symbols[i - 1].c, replacement), 1));
                    }
                }
                let new_s_len = new_s.len;
                self.symbols.insert(i, new_s); // Insert replacement before first char of pair
                self.symbols.remove(i + 1); // Remove first char of pair
                self.symbols.remove(i + 1); // And then the second

                // If there are other characters after the pair
                if i < self.symbols.len() - 1 {
                    changes.push(((second.c, self.symbols[i + 1].c), -1));
                    if self.symbols[i + 1].len + new_s_len < max_length {
                        changes.push(((replacement, self.symbols[i + 1].c), 1));
                    }
                }
            }

            i += 1;
        }

        changes
    }

    pub(super) fn remove(
        &mut self,
        removed_token: u32,
        split: &Vec<usize>,
        max_length: usize,
    ) -> Vec<(Pair, i32)> {
        //guaranteed that split.len()>1
        let mut changes: Vec<(Pair, i32)> = vec![];
        let mut i = 0;

        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Found a pair
            if self.symbols[i].c == removed_token {
                let removed_token_len = self.symbols[i].len;
                if i > 0 {
                    changes.push(((self.symbols[i - 1].c, removed_token), -1))
                }
                if i < self.symbols.len() - 1 {
                    changes.push(((removed_token, self.symbols[i + 1].c), -1));
                }
                let prev = self.symbols[i].prev;
                self.symbols.remove(i);
                for j in 0..split.len() {
                    let new_s = Symbol {
                        c: split[j] as u32,
                        prev: prev,
                        next: (i + j + 1) as isize,
                        len: removed_token_len, //get some len!!!!!
                    };
                    if (i > 0) && (self.symbols[i + j - 1].len + new_s.len < max_length) {
                        changes.push(((self.symbols[i + j - 1].c, split[j] as u32), 1));
                    }
                    self.symbols.insert(i + j, new_s);
                }

                // // If there are other characters after the pair
                if (i + split.len() < self.symbols.len())
                    && (self.symbols[i + split.len() - 1].len + self.symbols[i + split.len()].len
                        < max_length)
                {
                    changes.push((
                        (
                            self.symbols[i + split.len() - 1].c,
                            self.symbols[i + split.len()].c,
                        ),
                        1,
                    ));
                }
            }

            i += 1;
        }

        changes
    }

    pub fn merge_split_all(
        &mut self,
        merges: &HashMap<Pair, Vec<(u32, u32)>>,
        splits: &HashMap<u32, Vec<(u32, Vec<u32>)>>,
    ) {
        let mut queue = BinaryHeap::with_capacity(self.symbols.len());

        queue.extend(
            self.symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, window)| {
                    let pair = (window[0].c, window[1].c);
                    merges
                        .get(&pair)
                        .and_then(|vec| vec.first())
                        .map(|&(rank, new_id)| {
                            Event::Merge(Merge {
                                pos: index,
                                rank,
                                new_id,
                            })
                        })
                }),
        );

        while let Some(top) = queue.pop() {
            match top {
                Event::Merge(event) => {
                    if self.symbols[event.pos].len == 0 {
                        continue;
                    }

                    if self.symbols[event.pos].next == -1 {
                        continue;
                    }

                    let next_pos = self.symbols[event.pos].next as usize;
                    let right = self.symbols[next_pos].clone();

                    let target_new_pair = (self.symbols[event.pos].c, right.c);
                    if !merges
                        .get(&target_new_pair)
                        .and_then(|vec| vec.iter().find(|&&(rank, _)| rank == event.rank))
                        .map_or(false, |(_, new_id)| *new_id == event.new_id)
                    {
                        continue;
                    }

                    self.symbols[event.pos].merge_with(&right, event.new_id);
                    self.symbols[next_pos].len = 0;

                    if right.next > -1 && (right.next as usize) < self.symbols.len() {
                        self.symbols[right.next as usize].prev = event.pos as isize;
                    }

                    let current = &self.symbols[event.pos];
                    if current.prev >= 0 {
                        let prev = current.prev as usize;
                        let prev_symbol = self.symbols[prev].clone();
                        let new_pair = (prev_symbol.c, current.c);
                        if let Some((rank, new_id)) = merges
                            .get(&new_pair)
                            .and_then(|vec| vec.iter().find(|&&(rank, _)| rank > event.rank))
                        {
                            queue.push(Event::Merge(Merge {
                                pos: current.prev as usize,
                                rank: *rank,
                                new_id: *new_id,
                            }));
                        }
                    }

                    let next = current.next as usize;
                    if next < self.symbols.len() {
                        let next_symbol = self.symbols[next].clone();
                        let new_pair = (current.c, next_symbol.c);
                        if let Some((rank, new_id)) = merges
                            .get(&new_pair)
                            .and_then(|vec| vec.iter().find(|&&(rank, _)| rank > event.rank))
                        {
                            queue.push(Event::Merge(Merge {
                                pos: event.pos,
                                rank: *rank,
                                new_id: *new_id,
                            }));
                        }
                    }
                    if let Some((rank, split)) = splits
                        .get(&current.c)
                        .and_then(|vec| vec.iter().find(|&&(rank, _)| rank > event.rank))
                    {
                        queue.push(Event::Split(Split {
                            pos: event.pos,
                            rank: *rank,
                            split: split.clone(),
                        }));
                    }
                }
                Event::Split(event) => {
                    // Do split
                    if self.symbols[event.pos].len == 0 {
                        continue;
                    }

                    if self.symbols[event.pos].next == -1 {
                        continue;
                    }

                    // Make sure we are not processing an expired queue entry
                    if !splits
                        .get(&self.symbols[event.pos].c)
                        .and_then(|vec| vec.iter().find(|&&(rank, _)| rank == event.rank))
                        .map_or(false, |(_, split)| *split == event.split)
                    {
                        continue;
                    }
                    if event.split.len() + event.pos > self.symbols.len() {
                        continue;
                    }
                    let mut prev = self.symbols[event.pos].prev;
                    for (i, &new_c) in event.split.iter().enumerate() {
                        if event.pos + i < self.symbols.len() {
                            self.symbols[event.pos + i].c = new_c;
                            self.symbols[event.pos + i].prev = prev;
                            self.symbols[event.pos + i].len = 1;
                            prev = (event.pos + i) as isize;
                            self.symbols[event.pos + i].next =
                                if event.pos + i < self.symbols.len() - 1 {
                                    (event.pos + i + 1) as isize
                                } else {
                                    -1
                                };
                        } else {
                            break;
                        }
                    }
                    queue.extend(event.split.windows(2).enumerate().filter_map(
                        |(index, window)| {
                            let pair = (window[0], window[1]);
                            merges
                                .get(&pair)
                                .and_then(|vec| vec.iter().find(|&&(rank, _)| rank > event.rank))
                                .map(|&(rank, new_id)| {
                                    Event::Merge(Merge {
                                        pos: event.pos + index,
                                        rank,
                                        new_id,
                                    })
                                })
                        },
                    ));

                    if event.pos + event.split.len() < self.symbols.len() {
                        self.symbols[event.pos + event.split.len()].prev =
                            (event.pos + event.split.len() - 1) as isize;
                    }
                }
            }
        }

        self.symbols.retain(|s| s.len != 0);
    }

    pub(super) fn get_chars(&self) -> Vec<u32> {
        self.symbols.iter().map(|s| s.c).collect()
    }

    pub(super) fn get_chars_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.symbols.iter().map(|s| s.c)
    }

    pub(super) fn get_offsets_iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut pos = 0;
        self.symbols.iter().map(move |symbol| {
            let new_pos = pos + symbol.len;
            let offset = (pos, new_pos);
            pos = new_pos;
            offset
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_merge_split_all() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3, 'll': 4, 'ell': 5, 'ell': 6}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // Define merges and splits
        let mut merges = HashMap::new();
        merges.insert((2, 2), vec![(0, 4)]); // 'll' has rank 0 and id 4
        merges.insert((1, 4), vec![(1, 5)]); // 'ell' has rank 1 and id 5

        let mut splits = HashMap::new();
        splits.insert(5, vec![(2, vec![1, 4])]); // has rank 2 and 'ell' splits into 'e' and 'll'

        // Perform merge_split_all
        word.merge_split_all(&merges, &splits);

        // The word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'
        merges.insert((1, 4), vec![(1, 6)]); // 'ell' has rank 3 and id 6
        word.merge_split_all(&merges, &splits);

        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                6u32, // 'e'
                3u32, // 'o'
            ]
        );
    }

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let changes = word.merge(2, 2, 4, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((2u32, 2u32), -1i32), // todo
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }

    #[test]
    fn test_remove() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3, 'll':4, 'he': 5, 'ell':6}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        word.merge(2, 2, 4, usize::MAX);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );
        let changes = word.remove(4, &vec![2, 2], usize::MAX);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                2u32, // 'l'
                2u32, // 'l'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This remove affects the counts for the pairs
        // ('e', 'll') ~= (1, 4) -1,
        // ('ll', 'o') ~= (4, 3) -1,
        // ('e', 'l') ~= (1, 2) + 1,
        // ('l', 'l') ~= (2, 2) + 1,
        // ('l', 'o') ~= (2, 3) + 1
        // .
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 4u32), -1i32), // count for ('e', 'll') should be decreased by 1.
                ((4u32, 3u32), -1i32), // count for ('ll', 'o') should be decreased by 1.
                ((1u32, 2u32), 1i32),  // count for ('e', 'l') should be increased by 1.
                ((2u32, 2u32), 1i32),  // count for ('l', 'l') should be increased by 1.
                ((2u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }
}
