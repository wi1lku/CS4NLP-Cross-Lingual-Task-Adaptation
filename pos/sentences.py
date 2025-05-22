


def get_datapoint(conllu_sentence):
    sentence_list = []
    label_list = []
    for word in conllu_sentence:
        sentence_list.append(f"{word['form']}")
        label_list.append(f"{word['form']}/{"CONTR" if word['upos'] == '_' else word['upos']}")
    sentence = " ".join(sentence_list)
    label = " ".join(label_list)
    
    return {
        "sentence": sentence,
        "label": label,
    }

def get_prompt(sentence):
    dp = get_datapoint(sentence)
    return f"""
    ===Input===
    {dp['sentence']}
    ===Output===
    """

def get_label(sentence):
    dp = get_datapoint(sentence)
    return f"""
    {dp['label']}
    """

def get_full(sentence):
    return f"""
    {get_prompt(sentence)}
    {get_label(sentence)}
    """