from echo_embeddings import EchoEmbeddingsMistral, EchoPooling, EchoParser
import torch

# These are the templates for the model.
# Tips:
# - Always include a beginning of sentence <s> (it isn't added for you!)
# - The parser will replace variables and compute embeddings on things inside of braces, 
#   so be sure to reference variables inside of braces only (e.g. {!%%prompt%%,} will be 
#   replaced with the prompt, and {%%text%%} will be replaced with the text)
# - The pooling will take the {mean, last} of the token embeddings that are inside braces
#   except when the braces start with {! which means the text won't be included}. See usage
#   in the example below.
# - Example: "<s>The last-token of {this text %%text%% will be </s>} even though there
#             is {!text after it.</s>}"
# - When using max_tokens, the parser will enforce that every separate {} has at most 
#   max_tokens; this means that if you have multiple braces, the max_tokens will be
#   enforced for each set of braces separately. This is why {</s>} is enclosed in 
#   separate braces: so that </s> will not be cut off if %%text%% exceeds the max_tokens.
templates = {
    'query': '<s>Instruct:{!%%prompt%%,}\nQuery:{!%%text%%}\nQuery again:{%%text%%}{</s>}',
    'document': '<s>Document:{!%%text%%}\nDocument again:{%%text%%}{</s>}',
}

# Create the model
path_to_model = 'jspringer/echo-mistral-7b-instruct-lasttoken'
model = EchoEmbeddingsMistral.from_pretrained(path_to_model)
model = model.eval()

# Create the parser
parser = EchoParser(path_to_model, templates, max_length=300)

# Create the pooling: strategy can either be mean or last
pooling = EchoPooling(strategy='last')

# specify the prompt, queries, and documents
prompt = 'Retrieve passages that answer the question'
queries = [
    'What is the capital of France?',
    'What is the capital of Deutschland?',
]
documents = [
    'Paris is the capital of France.',
    'Berlin is the capital of Germany.',
]

query_variables = [{'prompt': prompt, 'text': q} for q in queries]
document_variables = [{'text': d} for d in documents]

query_tagged = [('query', q) for q in query_variables]
document_tagged = [('document', d) for d in document_variables]

# Get the tokenized embeddings
with torch.no_grad():
    query_embeddings = pooling(model(parser(query_tagged)))['sentence_embedding']
    document_embeddings = pooling(model(parser(document_tagged)))['sentence_embedding']

# compute the cosine similarity
sim = lambda x, y: torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

print('Similarity between the queries and documents:')
for i, q in enumerate(queries):
    for j, d in enumerate(documents):
        similarity_score = sim(query_embeddings[i], document_embeddings[j])
        print('Computing similarity between:')
        print(f'  - {q}')
        print(f'  - {d}')
        print(f'  Cosine similarity: {similarity_score:.4f}')
