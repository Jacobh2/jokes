"""

"""

def load_latest_qustions(clazz):
    # Query for the latest 10 questions
    qas = clazz.query.order_by(clazz.id.desc()).limit(10).all()
    
    if not qas:
        return None

    # Format the list according to the template
    def format(qa):
        return qa.id, qa.created.strftime("%Y-%m-%d %H:%M"), qa.question, qa.answer, qa.vote

    return list(map(format, qas))
