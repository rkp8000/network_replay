from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class RidgeSearcher(Base):
    
    __tablename__ = 'ridge_searcher'
    
    id = Column(Integer, primary_key=True)
    
    smln_id = Column(String)
    role = Column(String)
    last_active = Column(DateTime)
    error = Column(String)
    traceback = Column(String)
    
    commit = Column(String)
    
    
class RidgeTrial(Base):
    
    __tablename__ = 'ridge_trial'
    
    id = Column(Integer, primary_key=True)
    
    searcher_id = Column(Integer, ForeignKey('ridge_searcher.id'))
    searcher = relationship('RidgeSearcher', backref='trials')
    
    seed = Column(Integer)
    
    # parameters
    ridge_h = Column(Float)
    ridge_w = Column(Float)
    p_inh = Column(Float)
    rho_pc = Column(Float)
    
    z_pc = Column(Float)
    l_pc = Column(Float)
    w_a_pc_pc = Column(Float)
    
    p_a_inh_pc = Column(Float)
    w_a_inh_pc = Column(Float)
    
    p_g_pc_inh = Column(Float)
    w_g_pc_inh = Column(Float)
    
    fr_ec = Column(Float)
    
    # results
    stability = Column(Float)
    activity = Column(Float)
    speed = Column(Float)


class EmbeddedSearcher(Base):
    
    __tablename__ = 'embedded_searcher'
    
    id = Column(Integer, primary_key=True)
    
    smln_id = Column(String)
    role = Column(String)
    last_active = Column(DateTime)
    error = Column(String)
    traceback = Column(String)
    
    commit = Column(String)
    
    
class EmbeddedTrial(Base):
    
    __tablename__ = 'embedded_trial'
    
    id = Column(Integer, primary_key=True)
    
    searcher_id = Column(Integer, ForeignKey('embedded_searcher.id'))
    searcher = relationship('EmbeddedSearcher', backref='trials')
    
    seed = Column(Integer)
    
    # parameters
    area_h = Column(Float)
    area_w = Column(Float)
    ridge_y = Column(Float)
    p_inh = Column(Float)
    rho_pc = Column(Float)
    
    z_pc = Column(Float)
    l_pc = Column(Float)
    w_a_pc_pc = Column(Float)
    
    p_a_inh_pc = Column(Float)
    w_a_inh_pc = Column(Float)
    
    p_g_pc_inh = Column(Float)
    w_g_pc_inh = Column(Float)
    
    fr_ec = Column(Float)
    
    # results
    stability = Column(Float)
    angle = Column(Float)
    activity = Column(Float)
    speed = Column(Float)
