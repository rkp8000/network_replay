from sqlalchemy import Column, ForeignKey
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class LinRidgeSearcher(Base):
    
    __tablename__ = 'lin_ridge_searcher'
    
    id = Column(Integer, primary_key=True)
    
    smln_id = Column(String)
    role = Column(String)
    last_active = Column(DateTime)
    error = Column(String)
    traceback = Column(String)
    
    commit = Column(String)
    
    
class LinRidgeTrial(Base):
    
    __tablename__ = 'lin_ridge_trial'
    
    id = Column(Integer, primary_key=True)
    
    searcher_id = Column(Integer, ForeignKey('lin_ridge_searcher.id'))
    searcher = relationship('LinRidgeSearcher', backref='trials')
    
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

    
class LinRidgeFullTrial(Base):
    
    __tablename__ = 'lin_ridge_full_trial'
    
    id = Column(Integer, primary_key=True)
    
    lin_ridge_trial_id = Column(Integer, ForeignKey('lin_ridge_trial.id'))
    lin_ridge_trial = relationship('LinRidgeTrial', backref='full_trials')
    
    commit = Column(String)
    seed = Column(Integer)
    
    replay_fr = Column(Float)
    replay_fr_min = Column(Float)
    replay_fr_max = Column(Float)
