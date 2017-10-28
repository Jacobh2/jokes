"""

"""

def load_latest_qustions(clazz):
    # Query for the latest 10 questions
    qas = clazz.query.order_by(clazz.id.desc()).limit(10).all()
    
    if not qas:
        return None

    # Format the list according to the template
    return [(qa.question, qa.answer, qa.vote) for qa in qas]
