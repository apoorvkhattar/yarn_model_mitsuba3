import drjit as dr
import mitsuba as mi
import numpy as np
    
import tqdm

variant = 'llvm_ad_rgb'

mi.set_variant(variant)

def get_length(point_list):
    length_list = np.zeros((point_list.shape[0], 3))
    for i in range(point_list.shape[0]):
        if i > 0:
            length_list[i,0] = length_list[i-1,0] + np.linalg.norm(point_list[i]-point_list[i-1])
            length_list[i,1] = length_list[i-1,1] + np.linalg.norm(point_list[i]-point_list[i-1])
            length_list[i,2] = length_list[i-1,2] + np.linalg.norm(point_list[i]-point_list[i-1])
    return length_list

def get_length_sum(point_list):
    length_sum = 0
    for i in range(point_list.shape[1]):
        if i > 0:
            length_sum += np.linalg.norm(point_list[0,i]-point_list[0,i-1])
    return length_sum

class Frame:
    def __init__(self, reference, tangent):
        self.t = tangent
        self.r = reference
        
        proj_r_on_t = self.t * (np.dot(self.r, self.t) / np.linalg.norm(self.r))
        
        self.r = self.r - proj_r_on_t
        self.r = self.r / np.linalg.norm(self.r)
        
        self.s = np.cross(self.t, self.r)
        self.s = self.s / np.linalg.norm(self.s)
        
def generateFrames(points, tangents, firstFrame):
    num_points = points.shape[0]
    
    frames = [firstFrame]
    
    for i in range(num_points - 1):
        v1 = points[i + 1] - points[i]
        c1 = np.dot(v1, v1)
        
        ref_L_i = frames[i].r - 2 / c1 * np.dot(v1, frames[i].r) * v1
        tang_L_i = frames[i].t - 2 / c1 * np.dot(v1, frames[i].t) * v1
        
        v2 = tangents[i+1] - tang_L_i
        c2 = np.dot(v2, v2)
        
        ref_next = ref_L_i - 2 / c2 * np.dot(v2, ref_L_i) * v2
        
        frames.append(Frame(ref_next, tangents[i+1]))
    
    return frames

def squareAngularSpeedMinimizationFunc(t, maxAngle, twistCount=0):
    return t * (maxAngle + twistCount * 2 * np.pi)

def adjustFramesWithBoundaryCondition(frames, firstFrame, lastFrame, parameterValues, adjustmentFunc):
    maxAngle = np.arccos(np.clip(np.dot(frames[-1].t, lastFrame.t), -1.0, 1.0))
    adjusted_frames = rotateFrames(frames, parameterValues, adjustmentFunc(parameterValues, maxAngle))
    return adjusted_frames

def rotateFrames(frames, parameterValues, rotationFunc):
    rotatedFrames = []
    
    for i in range(len(frames)):
        angle = rotationFunc[i]
        r = R.from_rotvec(angle * frames[i].t)
        r_pert = r.apply(frames[i].r)
        
        rotatedFrames.append(Frame(r_pert, frames[i].t))
        
    return rotatedFrames

def generateReferenceNormals(points_array):
    normal_list = np.zeros(points_array.shape)

    for i in range(points_array.shape[0]):
        tangents = np.zeros(points_array[0].shape)
        
        ''' Tangent Determination (Speed Can Be Improved)'''
        for j in range(points_array.shape[1]):
            if j == 0:
                v1 = mi.Vector3f(points_array[i,j+1] - points_array[i,j])
                v2 = mi.Vector3f(0)
                v = dr.normalize(v1)
            elif j == points_array.shape[1]-1:
                v1 = mi.Vector3f(points_array[i,j] - points_array[i,j-1])
                v2 = mi.Vector3f(0)
                v = dr.normalize(v1)
            else:
                v1 = mi.Vector3f(0)
                v2 = mi.Vector3f(points_array[i,j] - points_array[i,j-1])
                v = dr.normalize(v2)
            tangents[j] = v

        boundaryConditions = np.array(((0,1,-1), (0,0,1)))
        firstFrame = Frame(boundaryConditions[0], tangents[0])
        frames = generateFrames(points_array[i], tangents, firstFrame)
        lastFrame = Frame(boundaryConditions[1], tangents[-1])
        # frames = adjustFramesWithBoundaryCondition(frames, firstFrame, lastFrame, t, squareAngularSpeedMinimizationFunc)
        
        normals = []
        for j in range(points_array[i].shape[0]):
            normals.append(frames[j].r)
        
        normal_list[i] = np.array(normals)

    return normal_list

def generateRMF(point_list, tangents):
    boundaryConditions = np.array(((0,1,-1), (0,0,1)))
    firstFrame = Frame(boundaryConditions[0], tangents[0])
    frames = generateFrames(point_list, tangents, firstFrame)
    lastFrame = Frame(boundaryConditions[1], tangents[-1])
    # frames = adjustFramesWithBoundaryCondition(frames, firstFrame, lastFrame, t, squareAngularSpeedMinimizationFunc)
    
    normals = []
    for j in range(point_list.shape[0]):
        normals.append(frames[j].r)
    
    return normals