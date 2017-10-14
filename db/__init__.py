"""
Basic database functions.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .d_models import Base

import LOCAL as L


def make_session():
    """
    Connect to the database and return a new session object for that database.
    
    :return: session object
    """
    # build connection url from input
    try:
        user = L.POSTGRES_USER
        pw = L.POSTGRES_PW
        db = L.POSTGRES_DB
    except:
        raise NameError(
            'Specify user, pw, and db in "LOCAL.py".')

    url = 'postgres://{}:{}@/{}'.format(user, pw, db)

    # make and connect an engine
    engine = create_engine(url)
    engine.connect()

    # create all tables defined in d_models.py
    Base.metadata.create_all(engine)

    # get a new session
    session = sessionmaker(bind=engine)()

    return session
