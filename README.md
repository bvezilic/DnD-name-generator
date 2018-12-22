# DnD-name-generator

Greetings fellow D&D players! I have one question for you. What is the hardest decision you have to make when creating a new character? Exactly! Choosing a name. While there is a lot of online name generators that enables you to customize your character name by looking at your class, origin, birthplace, etc. we are not doing any of that fancy stuff here. Instead of having database dictating our beloved future character name, we are putting our faith in (trustworthy) hands of a neural network.

**Splendid! How are we going to do that?**

Well, firstly, we are going to get as much fantasy names possible. These will be obtained from D&D 5e Player's Handbook and Xanathar's Guide to Everything. Each name is going to be associated with the corresponding `race` and `gender`. From races, we have **Humans**, **Elves**, **Dwarfs**, **Half-orcs**, **Halflings**, **Tieflings** and **Dragonborns**. Sorry, Aarakocra and others for not making the cut! From genders, we'll only have **Male** and **Female** option. Apologies to other genders... Continuing, once we are set-up with data, we'll create a character-based RNN network, LSTM to be more precise (it's all about that memory lane, am I right?). Similarly to the language models we are going to train RNN to predict the next letter in a name given the previous letter, race and gender. After the model is trained, we can generate new names by giving the model a first letter (or a random one), race, gender and sample letters successively until we hit the stop character or reach some maximum word length.


**What a minute, this seems quite basic. What's the deal here...?**

True. This project is quite simple and heavily inspired by one of the examples in [PyTorch tutorials](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html). The main idea of this exercise is to examine how PyTorch work with generating new samples using recurrent models, handling of sequnces with variable lengths, data preparation and loading, accurate calculation of loss for padded time-steps, etc. But also do it in a fun way!

### Overview

This is section will cover some of the issues I've encountered during implementation and how these problems can be resolved within PyTorch. Naturally, comments will be more technically oriented because of this and hopefully PyTorch won't change that much in it's future versions to make this irrelevant.

#### collate_fn and pad_sequence
If you haven't had an experience with Datasets and DataLoader, the way they work is you create your own Dataset with samples, then, you assign that Dataset to DataLoader witch is going to be responsible for creating batch of samples. The issue that occurs here is that by default DataLoader will try to stack samples together. This will unfortunatly fail, as samples are vectors with different lengths, causing and error. To resolve this, we can define our own `collate_fn` which will be responsible for creating a batch tensor. See example below: 

```python
    def collate_fn(batch):
        """
        Prepares batch for model by sorting, concatenating and padding inputs.

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

#### loss ignore_index