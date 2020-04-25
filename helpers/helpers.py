# Write data to file
def write_data(tech_id, tech_name, sentence, source, date_crawled):
    with open('PDF_data.txt', 'a') as f:
        # text = match["tid"] + '\n' + match["name"] + '\n' + sent + '\n' + source + '\n' + date_crawled + '\n\n'
        text = tech_id + '\n' + t_name + '\n' + sentence + '\n' + source + '\n' + date_crawled + '\n\n'
        f.write(text)

def preprocess_sents(sentences_list):
    prepocessed_sents = []
    for s in sentences_list:
        # Replace any new lines separating parts of the sentence
        s = s.replace('\n', ' ')

        # Replace any double spaces which might result from previous step with a single space
        s = s.replace('  ', ' ')

        # Do a length check to skip empty strings and random punctuation
        if len(s) < 3:
            continue
        prepocessed_sents.append(s)
    return prepocessed_sents