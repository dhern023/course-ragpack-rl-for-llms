"""
Pytorchified version of https://github.com/StatQuest/RLHF

NOTE: num_tokens was changed to num_embeddings, as it's actually the vocabulary size
    similarly, dim_model should have been renamed to embedding size, but it's fine.
NOTE: original code assumed batch_size of 1, and we don't so for inference we use [1, sequence]
"""

import torch.nn
import lightning

import torch
import torch.nn
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    """
    angle = token position / rotation frequency
    (x,y) = (cos(angle), sin(angle))
    """
    def __init__(self, d_model, max_sequence_length=512):
        super().__init__()

        # column vector of word positions: [0, 1, 2, ..., max_len-1]
        token_positions = torch.arange(max_sequence_length).reshape(-1, 1) # (max_sequence_length, 1)

        # The 'rotation speeds' for each dimension pair.
        # We only need half the d_model because (Sine, Cosine).
        dimension_indices = torch.arange(0, d_model, 2) # (
        rotation_frequencies = torch.pow(10000, dimension_indices / d_model) # (d_model / 2)

        # Create the 'Angle Matrix' via an outer product.
        # Every cell [pos, dim] is a specific angle for that word at that feature.
        angular_coordinates = token_positions / rotation_frequencies # (max_sequence_length, d_model / 2)

        # We interleave Sine and Cosine to create (x, y) coordinate pairs.
        coordinate_map = torch.zeros(max_sequence_length, d_model) # (max_sequence_length, d_model)
        coordinate_map[:, 0::2] = torch.sin(angular_coordinates) # Fill in even columns
        coordinate_map[:, 1::2] = torch.cos(angular_coordinates) # Fill in odd columns

        # Save as a buffer so it's not a "parameter" but moves with the model
        self.register_buffer('positional_map', coordinate_map.unsqueeze(0))

    def forward(self, word_embeddings):
        """
        word_embeddings: [Batch, Current_Sequence_Length, d_model]
        Returns: The sum of Word Meaning (Embeddings) + Positional Coordinates.
        """
        current_length = word_embeddings.size(1)

        # Matrix Addition
        return word_embeddings + self.positional_map[:, :current_length, :]

class TransformerBody(torch.nn.Module):
    """
    Base class the Pre-trained and Reward model will use in the composition

    Embedding -> Transformer Blocks (self-attention + FFN) -> Hidden states
    NOTE: num_tokens was renamed to num_embeddings = VOCAB_SIZE to better reflect the terminology
    """
    def __init__(self, num_embeddings, dim_model, max_sequence_length, n_layers, n_heads):
        """
        vocab_size = num_embeddings
        context_window = max_sequence_length
        """
        super().__init__()
        self.embedding_tokens = torch.nn.Embedding(num_embeddings, dim_model)
        self.positional_encoder = PositionalEncoding(dim_model, max_sequence_length) # <--- The "Dials"
        # Simplified: Replace with your actual TransformerBlock implementation
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=dim_model, 
                nhead=n_heads, 
                dim_feedforward=dim_model, 
                dropout=0.0, 
                batch_first=True, 
                norm_first=True
            )
            for _ in range(n_layers)
        ])

    def forward(self, token_ids):
        """
        hard-code the mask to be a decoder
        """
        x = token_ids
        x = self.embedding_tokens(x)
        x = self.positional_encoder(x) # words have 'coordinates'

        device = x.device
        seq_len = x.size(1)
        # Use the official PyTorch method to generate the causal mask
        # This creates a float mask with 0s and -inf
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)
        return x # Returns [batch, seq_len, dim_model]

class LanguageModel(lightning.LightningModule):
    """
    Transformer Body -> LM Head (projection to num_embeddings dimension)
    """
    def __init__(self, instance_body, num_embeddings):
        super().__init__()

        self.body = instance_body
        self.lm_head = torch.nn.Linear(instance_body.embedding_tokens.embedding_dim, num_embeddings)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids):
        hidden_states = self.body(input_ids)
        return self.lm_head(hidden_states) # [batch, seq_len, vocab_size]

    def configure_optimizers(self):
        """
        Required by Lightning
        TODO: Use AdamW
        """
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        """
        NOTE: Original code used input_tokens[0] and labels[0] to get the first sentence,
            and thus a [1, sequence_length, vocab_size],
            which is readily supported by CrossEntropy

        # input_tokens: ["What", "is", "this", "<EOS>", "pizza"]
        # labels:       ["is", "this", "<EOS>", "pizza", "<EOS>"]
        """
        input_tokens, labels = batch
        batch_size, sequence_length = input_tokens.shape

        logits = self.forward(input_tokens) # [batch_size, sequence_length, vocabulary_size]
        vocabulary_size = logits.size(-1)

        # Collapse the Batch and Sequence into a single dimension of Total Tokens
        # to get one prediction vector per target token.
        total_tokens = batch_size * sequence_length

        # 4. View as a 2D matrix of predictions and a 1D list of targets
        loss_input = logits.view(total_tokens, vocabulary_size)
        loss_targets = labels.view(total_tokens)

        # inputs and labels are pre-shifted ("What" -> "is"), index 0 of loss_input matches index 0 of loss_targets.
        loss = self.loss_function(loss_input, loss_targets)

        return loss

class RewardModel(lightning.LightningModule):
    """
    Same as Language Model, but scalar head and uses pairwise-ranking loss

    loss = mean(-log*sigmoid(r_better - r_worse))
    """
    def __init__(self, instance_body):
        super().__init__()
        self.body = instance_body
        self.reward_head = torch.nn.Linear(instance_body.embedding_tokens.embedding_dim, 1)

    def forward(self, input_ids):
        """
        Assumes
            input is of the form "prompt <EOS> good answer <EOS> bad answer <EOS>"
            input is already masked
        """
        hidden_states = self.body(input_ids) # [batch_size, sequence, d_model]

        # slice the <EOS> position to get the summary vector for the entire sentence
        last_token_hidden = hidden_states[:, -1, :] # [batch_size, d_model]
        logits = self.reward_head(last_token_hidden) # [batch_size, 1]

        return torch.einsum('br -> b', logits) # [batch_size]

    def configure_optimizers(self):
        """
        Required by Lightning
        TODO: Use AdamW
        """
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        """
        NOTE: The original code used only a single sentence per batch, so they didn't need to use mean()
        """
        input_tokens, labels = batch # collect [batch_size, prompt_length], [batch_size, combined_label_len]

        output_better, output_worse = labels.chunk(2, dim=1)

        input_better = torch.cat((input_tokens, output_better), dim=1) # [batch_size, prompt_length + label_length], [batch_size, sequence]
        input_worse = torch.cat((input_tokens, output_worse), dim=1) # [batch_size, prompt_length + label_length], [batch_size, sequence]

        reward_better = self.forward(input_better) # [batch_size], chosen rewards
        reward_worse = self.forward(input_worse) # [batch_size], rejected rewards

        # Pairwise Ranking Loss
        ## For details, see: https://youtu.be/qPN_XZcJf_s
        ## NOTE: reward_better and reward_worse are arrays with
        ##       scores for each token. We only want the score for the
        ##       last token
        loss = -F.logsigmoid(reward_better - reward_worse).mean()

        self.log("train_loss", loss)

        return loss


d_model = 4
if (d_model % 2 != 0):
    print("NOTE: Due to how position encoding is coded, d_model must be an even number.")

## We're also increasing the number of tokens our model can handle
max_length = 10

tokens = ['what',
          'is',
          'statquest',
          'awesome',
          'squatch',
          'eats',
          'pizza',
          'norm',
          '<EOS>',
          '<PAD>']
ids = list(range(len(tokens)))

token_to_id = dict(zip(tokens, ids))
id_to_token = dict(map(reversed, token_to_id.items()))

def tokens2ids(tokens):
    output = []
    for token in tokens.split():
        output.append(token_to_id[token])

    return output

def ids2tokens(ids):
    output = []
    for id in ids:
        output.append(id_to_token[id])

    return " ".join(output)

print(tokens2ids("what is statquest <EOS>"))
print(ids2tokens(tokens2ids("what is statquest <EOS>")))

# <prompt> <EOS> <answer> <EOS>
# pre-trained model has 
#   inputs tokens[:-1]
#   outputs tokens[1:]
list_prompt_answer_sentences = [
    "what is statquest <EOS> awesome <EOS>",
    "statquest is what <EOS> awesome <EOS>",
    "what is norm <EOS> awesome <EOS>",
    "what is squatch <EOS> awesome <EOS>",
    "norm is what <EOS> awesome <EOS>",
    "squatch is what <EOS> awesome <EOS>",
    "squatch eats what <EOS> pizza <EOS>"
]

list_inputs = []
list_outputs = []
for sentence in list_prompt_answer_sentences:
    list_tokens = sentence.split()
    list_inputs.append(" ".join(list_tokens[:-1]))
    list_outputs.append(" ".join(list_tokens[1:]))

pretrain_inputs = torch.tensor([tokens2ids(sentence) for sentence in list_inputs])
pretrain_labels = torch.tensor([tokens2ids(sentence) for sentence in list_outputs])

pretrain_dataset = torch.utils.data.TensorDataset(pretrain_inputs, pretrain_labels)
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset)

shared_engine = TransformerBody(num_embeddings=len(tokens), dim_model=d_model, max_sequence_length=max_length, n_layers=4, n_heads=1)
model = LanguageModel(instance_body=shared_engine, num_embeddings=len(tokens))

## Because tutorial involves creating a bunch of models
## and seeing what kind of output they generate in response
## to different prompts, we're writing a function to handle
## generating output from any model.

def generate_output(model, prompt):
    """
    Grab the logits for the last token, then get the softmax
    NOTE: Original function assumed [size_context, vocab_size], not [batch_size, size_context, vocab_size]
    NOTE: Assumes the model is setup to set the vocab_size to be the last dimension
    NOTE: We set the generation limit to the max_sequence_length from the PositionalEncoder
    input is passed to (T, size_embedding) and (T, size_embedding)
    """
    eos_token_id = token_to_id["<EOS>"]
    # If user passed (Time,), fixes it to (batch_size=1, Time)
    if prompt.dim() == 1:
        prompt = prompt[None, :] # [1, sequence_length]

    input_length = prompt.size(dim=1) # size_context

    for i in range(input_length, max_length): # hard-coded max_sequence_length
        tensor_input = prompt[:, -max_length:]
        logits = model(tensor_input) # log counts
        logits = logits[:,-1:,:] # last token in each sequence reduce to (batch_size, num_embeddings)
        predicted_id = logits.argmax(dim=-1) # assumes vocab_size is always the last dimension
        prompt = torch.cat((prompt, predicted_id), dim=1) # (batch, t) + (batch, 1)

        if (predicted_id == eos_token_id).all(): # if the prediction is <EOS>, then we are done
            break

    output = prompt[:, input_length:] # get predicted part (batch_size = 1, t)
    predicted_ids = output.squeeze().tolist() # shape (t,)

    print("Predicted Tokens:\n")
    print("\t", ids2tokens(predicted_ids))

## Now test out the transformer...
generate_output(model, torch.tensor(tokens2ids("what is statquest <EOS>")))

# Decoder model "borrows" the engine to train next-token prediction
trainer = lightning.Trainer(max_epochs=30, deterministic=True)
trainer.fit(model, train_dataloaders=pretrain_dataloader) # shared_engine contains decoder-trained weights:

generate_output(model, torch.tensor(tokens2ids("what is statquest <EOS>")))
generate_output(model, torch.tensor(tokens2ids("statquest is what <EOS>")))

# Reward model "borrows" the SAME engine to train scalar scores
# It automatically starts with the SFT knowledge, no need to copy weights
model_reward = RewardModel(instance_body=shared_engine)

## This is an example of a "better" response
## direct inference requires a 2D tensor
scores = model_reward(torch.tensor(tokens2ids("squatch eats what <EOS> pizza <EOS>")).view(1,-1))
scores[-1] # use the last score as the output from the reward model
scores = model_reward(torch.tensor(tokens2ids("squatch eats what <EOS> awesome <EOS>")).view(1,-1))
scores[-1]

rl_inputs = torch.tensor([tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>"),
                          tokens2ids("squatch eats what <EOS>")])


# Now let's create the reponses. We'll do this by concatonated a "better" response with a "worse" response. The "better" response comes first. Later, when we're in the `training_step()` we'll split these two responses apart.


rl_labels = torch.tensor([tokens2ids("pizza <EOS> what <EOS>"),
                          tokens2ids("pizza <EOS> is <EOS>"),
                          tokens2ids("pizza <EOS> statquest <EOS>"),
                          tokens2ids("pizza <EOS> squatch <EOS>"),
                          tokens2ids("pizza <EOS> eats <EOS>"),
                          tokens2ids("pizza <EOS> norm <EOS>"),
                          tokens2ids("pizza <EOS> awesome <EOS>")])


# Lastly, let's put the new dataset in a `DataLoader`.


## Now let's package everything up into a DataLoader...
rl_dataset = torch.utils.data.TensorDataset(rl_inputs, rl_labels)
rl_dataloader = torch.utils.data.DataLoader(rl_dataset)


# Now that we have the data in a `DataLoader`, we can use it to train the **Reward Model**.

## now train the model
trainer = lightning.Trainer(max_epochs=50, log_every_n_steps=2, deterministic=True)
trainer.fit(model_reward, train_dataloaders=rl_dataloader)

# Now let's see if the **Reward** model now gives the "better" response a higher score than the "worse" response.
reward_better = model_reward(torch.tensor(tokens2ids("squatch eats what <EOS> pizza <EOS>")).view(1,-1))
reward_better[-1]
reward_worse = model_reward(torch.tensor(tokens2ids("squatch eats what <EOS> awesome <EOS>")).view(1,-1))
reward_worse[-1]

# **NOTE:** We can also calculate the **Loss** by hand to see if these scores result in a **Loss** value that is close to 0...

## See what the loss is...
-F.logsigmoid(reward_better[-1] - reward_worse[-1])
# ...and we see that the **Loss** is super close to 0. In other words, the scores generated for the "better" and "worse" responses minimize the **Loss**.

# Now let's see how the **Reward Model** scores prompt/response pairs (with "better" and "worse" responses) for something it has never seen before...
## Now let's score an input/output pair that the Reward Model has never seen before...
## This is an example of a "better" response:
reward_better = model_reward(torch.tensor(tokens2ids("norm eats what <EOS> pizza <EOS>")).view(1,-1))
reward_better[-1]
## Now score another input/output pair that the Reward Model has never seen before...
## This is an example of a "worse" response:
reward_worse = model_reward(torch.tensor(tokens2ids("norm eats what <EOS> awesome <EOS>")).view(1,-1))
reward_worse[-1]

# # Train the original model with RLHF
# First, let's see what the original model generates when given a prompt it was not trained on.
generate_output(model, torch.tensor(tokens2ids("norm eats what <EOS>")))
trainer.fit(model_reward, preference_data)