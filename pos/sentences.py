


def get_datapoint(conllu_sentence):
    sentence = ""
    label = ""
    for word in conllu_sentence:
        sentence += f"{word['form']} "
        label += f"{word['form']}/{word['upos']} "

    
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