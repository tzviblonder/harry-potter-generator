# Harry Potter Generator
### Who wouldn't want to be the next J.K. Rowling?

#### Chances are you won't be, but this algorithm comes close. Using recurrent-type layers such as LSTM's and GRU's, this model takes the entirety of Harry Potter and "learns" it in order to produce new text of a similar style. It encodes the text (on a character level) in an 800-dimensional space and puts it through the recurrent layers.
#### One obstacle that text generation often faces is falling into a loop - algorithms tend to repeat a few words or character indefinitely. I solve this problem by having the model learn, instead of a deterministic distribution in which the most likely letter is always chosen, a stochastic probability distribution from which an output character is drawn. This allows for the model to use the log likelihood as the loss function, learning weights that maximize the likelihood of given outputs. The randomness in this type of output prevents the model from falling into a loop.
