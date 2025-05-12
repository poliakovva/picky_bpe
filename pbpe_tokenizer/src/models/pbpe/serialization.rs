use super::{super::OrderedVocabIter, Pair, PbpeBuilder, PBPE};
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashMap;

pub type MergeMap = HashMap<Pair, Vec<(u32, u32)>>;
pub type SplitMap = HashMap<u32, Vec<(u32, Vec<u32>)>>;

impl Serialize for PBPE {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("PBPE", 10)?;

        // Start by small fields
        model.serialize_field("type", "PBPE")?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("end_of_word_suffix", &self.end_of_word_suffix)?;
        model.serialize_field("fuse_unk", &self.fuse_unk)?;
        model.serialize_field("byte_fallback", &self.byte_fallback)?;
        model.serialize_field("ignore_merges", &self.ignore_merges)?;

        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);
        model.serialize_field("vocab", &ordered_vocab)?;
        let merges: HashMap<_, _> = self
            .merges
            .iter()
            .map(|(pair, vec)| {
                let pair_str = serde_json::to_string(pair).unwrap();
                (pair_str, vec.clone())
            })
            .collect();
        model.serialize_field("merges", &merges)?;

        let splits: HashMap<_, _> = self
            .splits
            .iter()
            .map(|(token_id, vec)| {
                let token_str = self.vocab_r[token_id].clone();
                (token_str, vec.clone())
            })
            .collect();
        model.serialize_field("splits", &splits)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for PBPE {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "PBPE",
            &[
                "type",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
                "ignore_merges",
                "vocab",
                "merges",
                "splits",
            ],
            PBPEVisitor,
        )
    }
}

struct PBPEVisitor;
impl<'de> Visitor<'de> for PBPEVisitor {
    type Value = PBPE;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct PBPE")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = PbpeBuilder::new();
        let mut vocab: Option<HashMap<String, u32>> = None;
        let mut merges = None;
        let mut splits = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token" => {
                    if let Some(unk) = map.next_value::<Option<String>>()? {
                        builder = builder.unk_token(unk);
                    }
                }
                "continuing_subword_prefix" => {
                    if let Some(prefix) = map.next_value::<Option<String>>()? {
                        builder = builder.continuing_subword_prefix(prefix);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(suffix) = map.next_value::<Option<String>>()? {
                        builder = builder.end_of_word_suffix(suffix);
                    }
                }
                "fuse_unk" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.fuse_unk(suffix);
                    }
                }
                "byte_fallback" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.byte_fallback(suffix);
                    }
                }
                "ignore_merges" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.ignore_merges(suffix);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                "merges" => merges = Some(map.next_value()?),
                "splits" => splits = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "PBPE" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"PBPE",
                        ))
                    }
                },
                _ => {}
            }
        }
        let vocab = vocab.ok_or_else(|| serde::de::Error::missing_field("vocab"))?;

        let merges: HashMap<String, Vec<(u32, u32)>> =
            merges.ok_or_else(|| serde::de::Error::missing_field("merges"))?;
        let merges: MergeMap = merges
            .into_iter()
            .map(|(pair_str, vec)| {
                let pair: Pair = serde_json::from_str(&pair_str).unwrap();
                (pair, vec)
            })
            .collect();

        let splits: HashMap<String, Vec<(u32, Vec<u32>)>> =
            splits.ok_or_else(|| serde::de::Error::missing_field("splits"))?;
        let splits: SplitMap = splits
            .into_iter()
            .map(|(token_str, vec)| {
                let token_id = match vocab.get(&token_str) {
                    Some(token_id) => *token_id,
                    None => return Err(serde::de::Error::custom("Token not found in vocab")),
                };
                Ok((token_id, vec))
            })
            .collect::<Result<_, _>>()?;
        builder = builder.vocab_and_merges_and_splits(vocab, merges, splits);
        Ok(builder.build().map_err(Error::custom)?)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::pbpe::Vocab;

    #[test]
    fn test_serialization() {
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let mut merges: HashMap<Pair, Vec<(u32, u32)>> = HashMap::new();
        let mut splits: HashMap<u32, Vec<(u32, Vec<u32>)>> = HashMap::new();
        merges.insert((1, 2), vec![(0, 3)]);
        splits.insert(3, vec![(1u32, vec![1, 2])]);

        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, merges, splits)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();

        let data = serde_json::to_string(&pbpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"PBPE","unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":{"[1,2]":[[0,3]]},"splits":{"ab":[[1,[1,2]]]}}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(pbpe, reconstructed);

        // With a space in the token
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b c d".into(), 2),
            ("ab c d".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let mut merges: HashMap<Pair, Vec<(u32, u32)>> = HashMap::new();
        let mut splits: HashMap<u32, Vec<(u32, Vec<u32>)>> = HashMap::new();
        merges.insert((1, 2), vec![(0, 3)]);
        splits.insert(3, vec![(1u32, vec![1, 2])]);
        let pbpe = PbpeBuilder::default()
            .vocab_and_merges_and_splits(vocab, merges, splits)
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();
        let data = serde_json::to_string(&pbpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"PBPE","unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":{"[1,2]":[[0,3]]},"splits":{"ab c d":[[1,[1,2]]]}}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(pbpe, reconstructed);
    }
}
