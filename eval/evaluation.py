"""
There are two evaluation methods defined here:
- Insertion distance in similar insertions
- Insertion distance in similar sentences
"""
# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

from sklearn.metrics.pairwise import cosine_similarity

import nltk
from evalManager import evalManager

def create_list(file_name):
    result = list()
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Add any preprocessing steps here
            result.append(line.strip().lower())
    return result


def similar_sentence_score(test_input, test_output, corpus, output, soft_matching=False):
    # compare sentences without tags in original dataset and test dataset
    test_sentences = create_list(test_input)
    test_output_sentences = create_list(test_output)
    corpus_sentences = create_list(corpus)
    output_sentences = create_list(output)

    test_embeddings = model.encode(test_sentences)
    corpus_embeddings = model.encode(corpus_sentences)

    total_score = 0

    for i, test_embedding in enumerate(test_embeddings):
        max_score = -1
        max_j = -1
        for j, corpus_embedding in enumerate(corpus_embeddings):
            css = cosine_similarity(
                test_embedding.reshape(1, -1), corpus_embedding.reshape(1, -1)
            )
            if css > max_score:
                max_score = css
                max_j = j

        # print(test_output_sentences[i])
        # print(output_sentences[max_j])
        # print(css)

        # Look at the POS of insertion bigram
        test_output_tags = nltk.pos_tag(test_output_sentences[i].split())
        output_tags = nltk.pos_tag(output_sentences[max_j].split())
        evaluator = evalManager(soft_matching=soft_matching)

        for i in range(0, len(output_tags) - 1, 1):
            bigram_match_count = 0
            bigram_count = 0

            for j in range(0, len(test_output_tags) - 1, 1):
                output_tags_bigram = (output_tags[i], output_tags[i + 1])
                test_output_tags_bigram = (test_output_tags[j], test_output_tags[j + 1])

                # Insertions in first place
                if (
                    "(" in output_tags_bigram[0][0]
                    and "(" in test_output_tags_bigram[0][0]
                ):
                    bigram_count += 1
                    # Check if POS of other component same
                    if output_tags_bigram[1][1] == test_output_tags_bigram[1][1]:
                        # Check if insertion same
                        '''
                        if output_tags_bigram[0][0] == test_output_tags_bigram[0][0]:
                            bigram_match_count += 1
                        '''
                        if evaluator.match(output_tags_bigram[0][0], test_output_tags_bigram[0][0]):
                            bigram_match_count += 1

                # Insertions in second place
                if (
                    "(" in output_tags_bigram[1][0]
                    and "(" in test_output_tags_bigram[1][0]
                ):
                    bigram_count += 1
                    # Check if POS of other component same
                    if output_tags_bigram[0][1] == test_output_tags_bigram[0][1]:
                        # Check if insertion same
                        '''
                        if output_tags_bigram[1][0] == test_output_tags_bigram[1][0]:
                            bigram_match_count += 1
                        '''
                        if evaluator.match(output_tags_bigram[1][0], test_output_tags_bigram[1][0]):
                            bigram_match_count += 1

            total_score += bigram_match_count 
        # print(test_output_tags)
        # print(output_tags)
        # print(bigram_match_count/bigram_count)

    return total_score / len(test_sentences)


def generate_all_corpus_bigrams(output_sentences):
    all_sentences = []

    for output_sentence in output_sentences:
        sentence = []
        output_tags = nltk.pos_tag(output_sentence.split())
        for i in range(0, len(output_tags) - 1, 1):
            sentence.append((output_tags[i], output_tags[i + 1]))
        all_sentences.append(sentence)
    return all_sentences


def evaluate_against_corpus_bigrams(test_output_bigram, all_corpus_bigrams):
    total_score = 0

    for sentence in all_corpus_bigrams:
        local_score = 0
        insertion_count = 0
        for corpus_bigram in sentence:
            # Check if the insertion is the same
            # First position
            if "(" in test_output_bigram[0][0] and "(" in corpus_bigram[0][0]:
                # Check if POS is the same
                insertion_count += 1
                if test_output_bigram[1][1] == corpus_bigram[1][1]:
                    local_score += 1

            # Second position
            if "(" in test_output_bigram[1][0] and "(" in corpus_bigram[1][0]:
                # Check if POS is the same
                insertion_count += 1
                if test_output_bigram[0][1] == corpus_bigram[0][1]:
                    local_score += 1

        total_score += local_score

    return total_score / len(all_corpus_bigrams)


def similar_insertion_score(test_output, output):
    test_output_sentences = create_list(test_output)
    output_sentences = create_list(output)

    all_corpus_bigrams = generate_all_corpus_bigrams(output_sentences)

    total_score = 0

    for test_output_sentence in test_output_sentences:
        test_output_tags = nltk.pos_tag(test_output_sentence.split())

        sentence_score = 0

        for i in range(0, len(test_output_tags) - 1, 1):
            test_output_bigram = (test_output_tags[i], test_output_tags[i + 1])
            local_score = evaluate_against_corpus_bigrams(
                test_output_bigram, all_corpus_bigrams
            )
            sentence_score += local_score

        total_score += sentence_score

    return total_score / len(test_output_sentences)


if __name__ == "__main__":
    nltk.download('averaged_perceptron_tagger')
    # Test input
    test_input = "../data/test_input.txt"
    # Model output
    test_output = "../data/test_output.txt"
    # Test input with actual insertions
    ground_truth = "../data/for_test_raw.txt"

    # Just sentences
    corpus = "../data/corpus.txt"
    # Sentences with insertions
    output = "../data/output.txt"

    sss_hard = similar_sentence_score(test_input, test_output, corpus, output, soft_matching=False)
    sss_soft = similar_sentence_score(test_input, test_output, corpus, output, soft_matching=True)
    sis = similar_insertion_score(test_output, output)

    print("Similar sentence(hard-matching) score on corpus:", sss_hard)
    print("Similar sentence(soft-matching) score on corpus:", sss_soft)
    print("Similar insertion score on corpus:", sis)
