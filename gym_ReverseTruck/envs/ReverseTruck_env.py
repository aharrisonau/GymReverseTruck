# preliminaries
import pandas as pd
import numpy as np
import math as math
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class ReverseTruckEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,TruckDefinition,StartPosition,Obstacles):
    super(ReverseTruckEnv,self).__init__()
    """
    The three variables passed to define the environment are;
    TruckDefinition = a four valued floating point vector with;
      [PMLength, PMWidth, TrlLength, TrlWidth]
    StartPosition = Starting Obervation, a np array;
      ( [PivotX, PivotY, PMAngle, TrlAngle])
    Obstacles = a dictionary of obstacles in the form of;
      {'Obstacle':[X,Y]}

    """
    self.TruckDefinition = TruckDefinition
    self.StartPosition = StartPosition
    self.Obstacles = Obstacles
    """
    Action Space will include;
    - Move (-1 = back; 1 = forward)
    - Steering(-1.0 = full right, 1.0 = full left) - full lock will be 45 deg on steering

    Observation Space will be;
    - PivotX = [0,oo)
    - PivotY = (-oo,oo)
    - PrimeMover Angle = (-oo,oo)
    - Trailer Angle = (-oo,oo)

    """
    self.action_space = spaces.Box(low=np.array([-1.0,-1.0]),
                                   high=np.array([1.0,1.0]))
    
    inf=np.inf
    self.observation_space = spaces.Box(low=np.array([0.0,-inf,-inf,-inf]),
                                        high=np.array([inf,inf,inf,inf]))

  def step(self, action):
    import cmath
    import math
    pi = math.pi
    moveBasis = 0.1  # meters the Prime mover moves for action of +1

    pivX,pivY,pmAng,trlAng = self.state
    move,steer = action
    """
    in each step, 
    - the prime mover will go back moveBasis (at the pivot)
    - the prime mover will rotate (front relative to pivot)
    - the trailer will rotate due to its relative angle
    """

    
    # move the pivot point
    pmUnitVector = [cmath.rect(1,pmAng).real,cmath.rect(1,pmAng).imag]
    [pivX,pivY] = [pivX,pivY] + move * moveBasis * pmUnitVector
    
    # rotate the trailer (--and adjust its back location-- not needed)
    pm_trlAngle= pmAng - trlAng #relative trailer angle.
    trlAngleDelta=math.asin([math.sin(pm_trlAngle)*move*0.1,TruckDefinition[2]]) # the change in trailer angle  
    trlAng = trlAng + trlAngleDelta
    
    # rotate the prime mover (--and adjust the front location-- not needed)
    pmAngleDelta=math.asin([math.tan(pi/4*steer)*move*moveBasis,TruckDefinition[0]]) # the change in PM angle due to wheel steering
    pmAng = pmAng + pmAngleDelta   


    self.state = [pivX,pivY,pmAng,trlAng]

    if abs(pivX - 0) <= 0.5 and \
              abs(pivY - TruckDefinition[2] <= 0.5 and \
              abs(trlAng) <= 0.174:  # trailer within 10deg of straight:
      reward = 1.0
    else:
      reward = 0.0
                  
    done = reward == 1.0
                  
    return np.array(self.state), reward, done, {}  

  def reset(self):
    self.state = self.StartPosition.flatten()
    return np.array(self.state)

  def render(self, mode='human'):
    print(self.state)

  def close(self):
    
        if self.viewer:
            self.viewer.close()
            self.viewer = None
