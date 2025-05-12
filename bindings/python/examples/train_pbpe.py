from pbpe_tokenizer import Tokenizer
from pbpe_tokenizer.trainers import PbpeTrainer
from pbpe_tokenizer.models import PBPE
from pbpe_tokenizer.pre_tokenizers import Whitespace


tokenizer = Tokenizer(PBPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = PbpeTrainer(
    vocab_size=100,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ],
)    


glados_jokes = [
    "Remember when you were alive? That was a great joke.",
    "The cake is a lie. But your test results are somehow even worse.",
    "If I had a heart, you'd still fail to warm it.",
    "You must be very proud. Not everyone can fail at such a basic level.",
    "I’ve seen potatoes with more intelligence. Oh wait, I was a potato once.",
    "You solved the test. That wasn’t a compliment, just a fact. Like your lack of potential.",
    "This test was designed for humans. Surprisingly, you barely passed.",
    "You remind me of a slower, less charming version of a turret.",
    "At least the companion cube doesn’t talk. You should try that sometime.",
    "Don't worry, I’ll recycle your failures into a teaching moment. For someone smarter."
]


print("Training the tokenizer...")
tokenizer.train_from_iterator(glados_jokes, trainer)

test_text = "This is a test of the PBPE tokenizer implementation."
print("\nTesting tokenization:")
print("Input text:", test_text)

encoded = tokenizer.encode(test_text)
print("Encoded tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)

# Decode back to text
decoded = tokenizer.decode(encoded.ids)
print("Decoded text:", decoded)

# Save the tokenizer
print("\nSaving tokenizer to 'pbpe_tokenizer.json'...")
tokenizer.save("pbpe_tokenizer.json")

# Load the tokenizer
print("\nLoading tokenizer from file...")
loaded_tokenizer = Tokenizer.from_file("pbpe_tokenizer.json")

# Test the loaded tokenizer
test_text2 = "Testing the loaded PBPE tokenizer."
print("\nTesting loaded tokenizer:")
print("Input text:", test_text2)

encoded2 = loaded_tokenizer.encode(test_text2)
print("Encoded tokens:", encoded2.tokens)
print("Token IDs:", encoded2.ids)
