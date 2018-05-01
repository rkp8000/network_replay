from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SmlnRslt(Base):
    
    __tablename__ = 'smln_rslt'
    
    id = Column(Integer, primary_key=True)
    group = Column(String)
    
    params = Column(JSONB)
    s_params = Column(JSONB)
    apxn = Column(Boolean)
    
    metrics = Column(JSONB)
    success = Column(Boolean)
    
    prep_time = Column(Float)
    run_time = Column(Float)
    
    ntwk_file = Column(String)
    smln_included = Column(Boolean)
    
    commit = Column(String)
 
    parent_id = Column(Integer, ForeignKey('smln_rslt.id'))
    parent = relationship('SmlnRslt', remote_side=[id])
    