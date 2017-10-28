"""

"""

def load_latest_qustions(clazz):
    return [('test1','answer1',1)]*100
    # Query for the latest 10 questions
    qas = clazz.query
    
    if not qas:
        return None

    # Format the list according to the template
    return [(qa.question, qa.answer, qa.vote) for qa in qas]
