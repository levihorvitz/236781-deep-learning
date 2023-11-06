r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=512,
        seq_len=64,
        h_dim=1024,
        n_layers=3,
        dropout=0.2,
        learn_rate=2e-3,
        lr_sched_factor=0.1,
        lr_sched_patience=3,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "You are the chosen one"
    temperature = 0.0005
    # ========================
    return start_seq, temperature


part1_q1 = r"""
1.  Training on the whole text, especially for large texts, can present a few challenges in the machine learning and particularly deep learning. Here are some reasons why we split the corpus into sequences:

•	Memory Constraints: In most modern deep learning architectures, the entire sequence needs to be loaded into memory at once for processing. If we use the whole text, the sequence can become so long that it exceeds the memory capacity of the hardware (such as GPU memory), especially for large datasets.

•	Batch Processing: By splitting the text into smaller sequences, we can process multiple sequences in parallel, which greatly speeds up training.

•	Learning Context: By breaking the text into smaller sequences, the model can learn from the local context around words and gradually understand the overall semantics of the text.

•	Backpropagation Through Time (BPTT): For large sequences, BPTT can become computationally expensive and more prone to errors due to the vanishing and exploding gradient problems.

"""

part1_q2 = r"""
2.  This 'longer memory' capability comes from two main factors:

1.	Overlapping sequences during training: During the training process, the model processes many overlapping sequences. For instance, after learning from the sequence ['The', 'cat', 'sat', 'on', 'the'], it would next learn from ['cat', 'sat', 'on', 'the', 'rug']. This overlap allows the model to learn associations and dependencies between elements that are technically more than a single sequence length apart, which contributes to the model's effective memory when generating text.

2.	Model architecture's memory mechanisms: Certain types of sequence models, like LSTMs and Transformers, are designed with mechanisms to 'remember' or give importance to critical information from earlier in the sequence. For example, LSTMs have a cell state that carries forward relevant information through time, while Transformers use an attention mechanism to focus on important parts of the input sequence. These mechanisms provide the models with an effective 'memory' that spans beyond the immediate input sequence length.
"""

part1_q3 = r"""
3.  When dealing with sequential data, such as time series or textual data, we typically avoid shuffling the order of the batches during the training process. This practice is primarily due to the inherent nature of sequence models like RNN, LSTM, or Transformers, which are designed to leverage the order of data points for effective learning.
Shuffling the data could disrupt the temporal relationships that these models strive to learn. For instance, in a text sequence, the contextual meaning of a word often heavily relies on the preceding words. Shuffling the data would reorganize the word order, thereby rendering the model incapable of learning these crucial dependencies.
"""

part1_q4 = r"""
4.  The 'temperature' in the context of a neural network, such as an RNN used for text generation, is a parameter that controls the randomness of the predictions.
•	High Temperature: If the temperature is high (e.g., much greater than 1.0), it makes the probability distribution closer to uniform. This means all characters (or whatever the output space is) have almost equal probability of being the next output. The output becomes more randomized and less predictable. This can result in more diverse and creative output, but it also tends to produce more mistakes and less coherent sequences.

•	Low Temperature: If the temperature is low (e.g., much less than 1.0), it makes the probability distribution sharper, increasing the likelihood of the most probable output. This tends to make the model more confident in its predictions but also more conservative. It will likely choose the most likely next character, resulting in more predictable and coherent output, but potentially less diverse and creative.

•	Temperature of 1.0: This is often the default and provides a balance between diversity/creativity and accuracy/predictability.

we lower the temperature for sampling to make the generated text more coherent and less random. While this might make the output less diverse and potentially less creative, it increases the chances of the output making sense and being grammatically correct. The ideal temperature often requires some experimentation and can depend on the specific task and the trained model.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0,
        z_dim=0,
        x_sigma2=0,
        learn_rate=0.0,
        betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        h_dim=64,
        z_dim=32,
        x_sigma2=0.001,
        learn_rate=0.0001,
        betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
1.  In the VAE, σ² would typically be the variance of the latent space distribution. Here's how different values of σ² might affect the VAE:
•	Low σ² values: A low variance means that the data points are very close to the mean and to each other, signifying a tight clustering of the encoded representations in the latent space. This could mean the VAE is very certain about the encoding of each data point, possibly leading to overfitting if the σ² is too low, as it might not generalize well to unseen data.
•	High σ² values: A high variance means the data points are spread out from the mean and from each other, indicating a broad dispersion of the encoded representations in the latent space. This could mean the VAE is less certain about the encoding of each data point, possibly leading to underfitting if the σ² is too high, as it might generate more diverse but less accurate outputs.
In the context of a VAE, the σ² parameter controls the level of noise in the VAE's latent space. If σ² is low, the noise level is low and the latent points are closer to their mean values. If σ² is high, the noise level is high and the latent points are more spread out around their mean values.
"""

part2_q2 = r"""
2.
1.	Purpose of the VAE loss terms:

•	Reconstruction loss: The first part of the VAE loss function is the reconstruction loss. This is used to measure how well the decoder is performing - that is, how closely the output of the decoder matches the original input. 

•	KL Divergence loss: The second part of the VAE loss function is the KL Divergence loss. This is used to measure how closely the distribution of the encoded representations (produced by the encoder) matches a target distribution, usually a standard normal distribution (mean = 0, variance = 1). The closer the encoded distribution is to the target distribution, the lower the KL Divergence loss.
2.	Effect of KL loss term on the latent-space distribution: 

The KL divergence loss term acts as a regularizer for the latent space distribution. Specifically, it tries to push the parameters of the latent space distribution (i.e., the mean and variance of the distribution that the encoder network outputs for a given input) towards the parameters of a standard normal distribution. As a result, the encoded representations are distributed in a manner that is more regular and well-behaved, which can make the latent space easier to interpret and manipulate, and the sampling process more stable.

3.	Benefits of the KL loss term effect: By ensuring that the encoded representations follow a standard normal distribution, we can easily generate new data by sampling from this distribution in the latent space. This makes the VAE a powerful generative model: once the model is trained, we can sample a point from the latent space, decode it, and get a brand new data point that is similar to the data the VAE was trained on. This wouldn't be as straightforward if the latent space distribution didn't follow a standard normal distribution, because we wouldn't know from where to sample. By enforcing this constraint, the KL divergence loss makes the VAE a more useful and versatile model for data generation.
"""

part2_q3 = r"""
3.  The term 'evidence' in the context of VAEs refers to the observed data, and p(X) represents the probability distribution of this observed data. The idea behind maximizing the evidence distribution, p(X), is essentially to find the most likely parameters of the model that explain the observed data. By doing this, we aim to construct a model that generates data very similar to the observed data.
However, in practice, directly maximizing p(X) is computationally challenging because it involves an intractable integral over the latent variables. Instead, VAEs maximize a lower bound on the log of p(X), also known as the evidence lower bound (ELBO). This is where the loss components of the VAE come into play.
"""

part2_q4 = r"""
4.  Modelling the logarithm of the variance in the encoder of a VAE instead of the variance itself has several advantages:

•	Stability: The space of variances is restricted to positive real numbers, but the space of log-variances spans the entire real line. This is easier to handle numerically and can help stabilize the learning process. 

•	Scale Invariance: Modelling log-variance also helps to handle scale differences in the data. Variance can span several orders of magnitude, and using log-variance can help mitigate this.

•	Better Gradients: In backpropagation, the gradients for variance might be very large if the variance is close to zero. By modelling log-variance, we can avoid this problem and get more balanced gradients, which can help the training process.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=0,
        num_heads=0,
        num_layers=0,
        hidden_dim=0,
        window_size=0,
        droupout=0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim = 128, 
        num_heads = 4,
        num_layers = 2,
        hidden_dim = 180,
        window_size = 128,
        droupout = 0.2,
        lr=0.0005,
    )
    # ========================
    return hypers


part3_q1 = r"""
1. When multiple encoder layers with sliding-window attention are stacked, each successive layer indirectly captures a larger context than the immediate window. This happens because the higher layer's attention window spans over multiple windows from the layer beneath it, each of which has already integrated their own local contexts. This mechanism is analogous to how in Convolutional Neural Networks, higher layers capture broader receptive fields by stacking local receptive fields from lower layers. Therefore, the final layer in a stacked encoder system can incorporate a broader context, enhancing the model's overall understanding of the input.
"""

part3_q2 = r"""
2. Mixed Attention for Global Context:
One possible approach to retain computational complexity similar to sliding-window attention O(nw) while computing attention on a more global context is to use a combination of local and global attention, often referred to as 'interleaved' or 'mixed' attention.
In this approach, two separate attention patterns could be utilized: local sliding-window attention and global attention. The local attention would function similarly to the typical sliding-window attention, focusing on a small, fixed-size window around the current word. The global attention, on the other hand, would operate at a reduced resolution, taking in larger strides across the sequence, essentially summarizing broader context into fewer steps.
To manage computational complexity, the global attention can be sparse, focusing only on certain key parts of the sequence rather than the whole sequence. This way, the model would still have a sense of the entire sequence but wouldn't have to perform computationally expensive operations for each and every token.
The outputs of local and global attention could be combined, for instance by concatenation followed by a linear transformation, to form the final attention output. This kind of mixed attention mechanism would provide a blend of local and global context while keeping the computational complexity in check.
"""


part4_q1 = r"""
Fine-tuning outperforms training from scratch when comparing it to part 3. This can be attributed to several reasons. Firstly, the model we fine-tuned is larger than the one built from scratch, giving it enhanced expressiveness. Additionally, during the pretraining phase, it was trained on significantly larger datasets. Moreover, the pretrained model was exposed to tasks similar to the downstream task, making the acquired context "relevant" and leading to better results when using only training of the classifiers. This observation holds true not only for NLP models like BERT but also for vision models, where contrastive learning has been found to outperform training from scratch.

However, it is important to note that these advantages of fine-tuning are typically observed when using a pretrained model from the same domain and on similar tasks. If the task is highly specific or fundamentally different in nature, or if the domain diverges significantly (such as using BERT for non-NLP time series classification), training from scratch may yield better results.
"""

part4_q2 = r"""
When considering fine-tuning, it is possible to utilize internal layers of the model. However, we suspect that although the model would be able to learn, the results would be inferior compared to fine-tuning the last layers. Similar to CNNs, as we delve deeper into the architecture, the learned representations become more intricate and task-specific. This implies that the middle layers (multihead attention blocks) capture general dependencies and relationships between words or tokens in the input sequence that can be applied to various tasks. On the other hand, the last layers (classification head) adapt these general representations to the specific task at hand.

Drawing an analogy from this observation, when fine-tuning an NLP task such as sentiment analysis, we can leverage the general representations as they are and focus on fine-tuning the last layers. Although there may be some improvement by enhancing the general representations and aligning them with the specific task, the benefits would be limited since minimal training is required for the middle layers (assuming the model was pretrained on the same domain) and the classification head remains unchanged. However, this scenario can change if we fine-tune for a task from a different domain, such as general time series analysis, where the middle layers might not adequately represent the dependencies of our dataset.
"""


# ==============
