import torch
from model.transformer import *
from model.tokenizer import *
import argparse
import tiktoken
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--N', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--vocab_size', type=int, default=1000)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--device', type=str, default='cuda')
cfg = vars(parser.parse_args())

num_epoch = cfg['num_epoch']
N = cfg['N']
batch_size = cfg['batch_size']
vocab_size = cfg['vocab_size']
seq_len = cfg['seq_len']
d_model = cfg['d_model']
d_ff = cfg['d_ff']
num_head = cfg['num_head']
dropout = cfg['dropout']
eps = cfg['eps']
device = cfg['device']

if __name__ == "__main__":
    model = Transformer(N, vocab_size, d_model, d_ff, num_head, dropout=dropout).to(device)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text = """Text-to-image models offer unprecedented freedom to guide creation through natural language. Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes. In other words, we ask: how can we use language-guided models to turn our cat into a painting, or imagine a new product based on our favorite toy? Here we present a simple approach that allows such creative freedom. Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model. These "words" can be composed into natural language sentences, guiding personalized creation in an intuitive way. Notably, we find evidence that a single word embedding is sufficient for capturing unique and varied concepts. We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.
Our code, data and new words will be available at: this https URL"""
    data = tokenizer.encode(text)
    mapper = TokenMapper(data)
    mapper.append(data)
    data = mapper.encode(data)
    data = torch.tensor(data, dtype=torch.int64, device=device)
    print(data.shape)
    for epoch in tqdm(range(num_epoch)):
        pass