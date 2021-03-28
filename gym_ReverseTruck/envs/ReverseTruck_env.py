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
    StartPosition = Starting Obervation, a 3x2 np array;
      ( [PMFrontX,PMFrontY],
        [PivotX,PivotY],
        [TrlBackX,TrlBackY])
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
    - PrimeMoverX = (-oo,oo)
    - PrimeMoverY = [0,oo)
    - PivotX = (-oo,oo)
    - PivotY = [0,oo)
    - TrailerBackX = (-oo,oo)
    - TrailerBackY = [0,oo)

    """
    self.action_space = spaces.Box(low=np.array([-1.0,-1.0]),
                                   high=np.array([1.0,1.0]))
    
    inf=np.inf
    self.observation_space = spaces.Box(low=np.array([-inf,0.0,-inf,0.0,-inf,0.0]),
                                        high=np.array([inf,inf,inf,inf,inf,inf]))
    
  def 2D_Vector_Rotation(self,2D_Vector,rotationAngle):
    import numpy as np
    rotationMatrix = np.array([[np.cos(rotationAngle),-np.sin(rotationAngle)],
                           [np.sin(rotationAngle),np.cos(rotationAngle)]])
    return np.matmul(rotationMatrix,2D_Vector)
   
  def step(self, action):
    pmX,pmY,pivX,pivY,trlX,trlY = self.state
    move,steer = action
    """
    in each step, 
    - the prime mover will go back 0.1m (at the pivot)
    - the prime mover will rotate (front relative to pivot)
    - the trailer back will rotate due to its relative angle
    """

    pmVect=np.array([pmX-pivX,pmY-pivY]) # prime mover vector
    trlVect=np.array([pivX-trlX,pivY-trlY]) # trailer vector
    pivotMovement=np.linalg.norm(pmVect)*move*0.1 # amount the pivot moves
    pmAngleDelta=math.asin([math.tan(math.pi/4*steer)*move*0.1,TruckDefinition[0]]) # the change in PM angle due to wheel steering
    pm_trlSin = np.cross(trlVect,pmVect,)/np.linalg.norm(pmVect)/np.linalg.norm(trlVect) #sine of the trailer angle relative to the PM
    pm_trlAngle=math.asin(np.clip(pm_trlSin,-1,1)) #relative trailer angle.  Clip eliminate rounding errors giving arguments >1
    trlAngleDelta=math.asin([math.sin(pm_trlAngle)*move*0.1,TruckDefinition[2]]) # the change in trailer angle
    
    # move the pivot point
    pmUnitVector = pmVect/np.linalg.norm(pmVect)
    [pivX,pivY] = [pivX,pivY] + move * 0.1 *pmUnitVector
    
    # rotate the prime mover and adjust the front location
    pmVect = 2D_Vector_Rotation(pmVect,pmAngleDelta)
    [pmX,pmY] = [pivX,pivY] + pmVect
    
    # rotate the trailer and adjust its back location
    trlVect = 2D_Vector_Rotation(trlVect,trlAngleDelta)
    [trlX,trlY] = [pivX,pivY] - trlVect
    self.state = pmX,pmY,pivX,pivY,trlX,trlY

    done = False

    reward = 0.0

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
