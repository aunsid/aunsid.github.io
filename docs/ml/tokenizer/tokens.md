# Tokens

Large language models (LLMs) cannot work with text in its raw form (strings). The text needs to be converted into an integer representation first. The process of breaking an input sequence down into smaller units so that it can be used by LLMs is called tokenization. There are various ways this can be done — characters, subwords, and words. Each method has its strengths and drawbacks.

### Characters
The easiest way to assign IDs to text is to break the text down into characters, where each character gets a unique token. For example, take the input sequence: *This is an example text.*

We break down each character: [T, h, i, s, ' ', i, s, ' ', a, n, ' ', e, x, a, m, p, l, e, ' ', t, e, x, t]

* T - 0
* h - 1
* i - 2
* s - 3
* ' ' - 4
* i - 2 # assigning i and s the same token ids as they have been identified before
* s - 3
* ' ' - 4 # seen before
* a - 5
* n - 6
* ' ' - 4 # seen before
* e - 7
* x - 8
* a - 5 # seen before
* m - 9
* p - 10
* l - 11
* e - 7 # seen before
* ' ' - 4 # seen before
* t - 12
* e - 7
* x - 8 # seen before
* t - 12 # seen before

Pros:
- Ability to handle unseen/new words.

Cons:
- Long sequences are computationally expensive. The attention mechanism is an O(n²) operation, so cost grows quadratically with sequence length.


### Words
Another way of breaking the text down into smaller units would be by breaking the text into words.
For *This is an example text.*, the input sequence would be broken into [This, is, an, example, text].

* This - 0
* is - 1
* an - 2
* example - 3
* text - 4

Pros:
 - Sequences are much smaller so it is not computationally expensive

Cons:
 - Cannot handle new words.
 


### Subwords
Subword tokenization as the name suggests uses subwords instead of whole words. The idea behind using subword tokenization is to find a middle ground between character level and word level tokenization. This leads to smaller vocabulary sizes as well as the ability to handle unseen words.




#### Byte-Pair Encoding
For every iteration:
    1. Identify the most common pair in the text
    2. Assign a new id to the pair. Store this in a lookup table
    3. Replace all the occurrences with the new id in the text
    4. Do this till you reach the desired vocab size

Decoding
    To restore the original text, look up each id in the vocabulary table to get its byte sequence, concatenate all byte sequences, then decode the result as UTF-8.



### Embeddings
Token IDs are straightforward — they are unique ids that each part of the sequence receives, similar to vectorization. They do not capture any semantic relationships between tokens.
On the other hand, embeddings are richer vector representations that provide semantic meaning to the tokens based on how commonly they are used together with, or in similar contexts to, other tokens. These embeddings are learnt by the LLM during training.


Workflow:
Text -> tokenids -> vector embeddings -> output logits over vocabs -> tokenids -> Text

The way text is processed in an LLM workflow: first, text is converted into token ids using a tokenizer. These token ids are then passed to the model for processing. The model computes overall embeddings based on the individual embeddings of the learned tokens. During output generation, the model provides the vector value/embedding for the next token over the entire vocabulary. We then select the most probable token and convert it back into text.
In auto-regressive models, output generation is an iterative process. The output token is appended to the input sequence and fed into the whole workflow again to generate the next token. This happens till we reach the maximum context length the model can produce or the model outputs a special token indicating the end of a sequence.



## References
1. CS336n
2. https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-tokens


