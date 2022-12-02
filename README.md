# Naturalization of Text: Inserting Disfluencies 

Parth Vipul Shah, Ryan Luu, Bowman Brown, Alfianto Widodo, Ira Deshmukh

Department of Computer Science, University of Southern California, 941 Bloom Walk, Los Angeles, CA 90089, USA

## Abstract

In this work, we have successfully implemented methods to transform raw text documents of spoken dialogue/speech into its more human natural-sounding version by augmenting these documents with disfluencies. We present two such methods to naturalize text. The first method is a bigram approach that uses the frequency of the occurrences of desired bigrams in the training data as a basis to insert filler words or pauses in the given input sentences. The second method uses a transformer to learn the most probable insert locations and disfluencies. The performance of each model was then measured using two automated scoring systems based off similar sentence score and similar insertion score.
