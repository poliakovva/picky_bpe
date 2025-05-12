import inspect
import pbpe_tokenizer

for name, member in inspect.getmembers(pbpe_tokenizer):
    print(f"{name}: {type(member)}")