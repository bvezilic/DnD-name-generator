# DnD-name-generator

Greetings fellow D&D players! I have one question for you. What is the hardest decision you have to make when creating a new character? Exactly! Choosing a name. While there is a lot of online name generators that enables you to customize your character name by looking at your class, origin, birthplace, etc. we are not doing any of that fancy stuff here. Instead of having database dictating our beloved future character name, we are putting our faith in (trustworthy) hands of a neural network.

**Splendid! How are we going to do that?**

Well, firstly, we are going to get as much fantasy names possible. These will be obtained from D&D 5e Player's Handbook and Xanathar's Guide to Everything. Each name is going to be associated with the corresponding `race` and `gender`. From races, we have **Humans**, **Elves**, **Dwarfs**, **Half-orcs**, **Halflings**, **Tieflings** and **Dragonborns**. Sorry, Aarakocra and others for not making the cut! From genders, we'll only have **Male** and **Female** option. Continuing, once we are set-up with data, we'll create a character-based RNN network, LSTM to be more precise (it's all about that memory lane, am I right?). Similarly to the language models we are going to train RNN to predict the next letter in a name given the previous letter, race and gender. After the model is trained, we can generate new names by giving the model a first letter (or a random one), race, gender and sample letters successively until we hit the stop character or reach some maximum word length.


**What a minute, this seems quite basic. What's the deal here...?**

True. This project is quite simple and heavily inspired by one of the examples in [PyTorch tutorials](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html). The main idea of this exercise is to examine how PyTorch work with generating new samples using recurrent models, handling of sequnces with variable lengths, data preparation and loading, accurate calculation of loss for padded time-steps, etc. But also do it in a fun way!

### Implementation decisions and errors

This is section will cover some of the issues I've encountered during implementation and how these problems can be resolved within PyTorch. Naturally, comments will be more technically oriented because of this and hopefully, PyTorch won't change that much in its future versions to make this irrelevant.

#### collate_fn and pad_sequence
If you haven't had an experience with Datasets and DataLoader, the way they work is you create your own Dataset with samples, then, you assign that Dataset to DataLoader witch is going to be responsible for creating a batch of samples. The issue that occurs here is that by default DataLoader will try to stack samples together. This will unfortunately fail, as samples are vectors with different lengths and trying to stack them together will produce something similar to the jagged arrays which are not supported as input. To resolve this, we can use `pad_sequence` to fill missing gaps in our soon to be matrix within `collate_fn` method. Besides taking care of padding, this method will also do all the transforming processes in order to prepare inputs for the model. See example below: 

```python
def collate_fn(batch):
    """
    Prepares batch for the model by sorting, concatenating and padding inputs.

    :param batch: list of tuples [(train, target), ...]
    :return:
        inputs: Tensor with shape (max_length, batch_size, input_size)
        targets: Tensor with shape (max_length, batch_size)
        lengths: Tensor with shape (batch_size)
    """
    batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)

    # Splits batch and concatenate input vectors
    inputs, targets = zip(*batch)
    inputs = [torch.cat([sample['name'], sample['race'], sample['gender']], 1) for sample in inputs]

    # Get list of lengths per sequence
    lengths = [input.shape[0] for input in inputs]
    lengths = torch.tensor(lengths)

    # Padding
    inputs = pad_sequence(inputs, padding_value=0)
    targets = pad_sequence(targets, padding_value=-1)  # Specific value to be ignored during loss computation

    return inputs, targets, lengths
```

#### pack_sequence

Now that the inputs are properly formed, let's look at how the forward pass can be defined with the usage of `pack_sequence`. Examine the example below:

> Be careful with `batch_first` arguments when working with `pad_sequence` and `pack_sequence`. Mixing these can cause really hard to debug situations.


```python
def forward(self, inputs, hx, cx, lengths):
    """
    :param inputs: Input tensor with shape (max_length, batch_size, input_size)
    :param hx: Previous hidden state tensor with shape (num_layers, batch_size, hidden_size)
    :param cx: Previous memory state tensor with shape (num_layers, batch_size, hidden_size)
    :param lengths: Tensor containing length for each sample in batch with shape (batch_size)
    :return:
        logits: Output from dense layer tensor with shape (max_length, batch_size, output_size)
        hx: Output hidden state tensor with shape (num_layers, batch_size, hidden_size)
        cx: Output memory state tensor with shape (num_layers, batch_size, hidden_size)
    """
    inputs = pack_padded_sequence(inputs, lengths=lengths)
    outputs, (h_n, c_n) = self.lstm(inputs, (hx, cx))
    pad_outputs, _ = pad_packed_sequence(outputs)
    logits = self.dense(self.dropout(pad_outputs))

    return logits, h_n, c_n
```

That's pretty much it, pretty straight-forward. 
1) Transform inputs into PackedSequence
2) Run through the model
3) Pad them back to Tensor
4) ...
5) Profit?

#### loss ignore_index

This is something I haven't seen a lot of people do, and I am a bit confused why that is... Having padded elements due to variable sequence length, it's necessary to be careful when computing loss. `CrossEntropyLoss` has no concept of time-steps or which are the real output and which are the padded elements. Remember when we padded targets with -1 value in `collate_fn`? The same value can now be used to identify which time-steps were padded and ignore them during the computation of loss. This can be achieved by simply adding padded value to loss function like: `nn.CrossEntropyLoss(ignore_index=-1)`


### Performance evaluation

Wow, this README has gotten quite fat... Okay, let's wrap it up with evaluation. There is a notebook that's going to show some actual and interesting results instead of wall-of-text explaining how I struggled with such simple task.

Anyhow, what you can find there is:
1) Statistics of the dataset
2) Training of the model
3) Generating samples during different stages of training and comparing them with the most similar names in the dataset

That's everything I wanted to share, I can't believe you come all the way to the last line... Well, thank you and may the dice ever be in your favor! (Or Tensor shapes...)
