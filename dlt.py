class Dlt(object):

  # constructor
  def __init__(self, orgs, corrs):
    self.orgs = orgs      # set of key points from first image set  [(), (), (), ...]
    self.corrs = corrs    # set of key points from second image set [(), (), (), ...]
  
  # compute the centroid of a set of key points from an image
  def average(self):
    sumOrgs = np.zeros(2)
    sumCorrs = np.zeros(2)
    
    for point in self.orgs:
      sumOrgs[0] += point[0] # sum of the x-coordinates of the key points in the first image set
      sumOrgs[1] += point[1] # sum of the y-coordinates of ...                   second ...
    
    for point in self.corrs:
      sumCorrs[0] += point[0] # sum of the x-coordinates of the key points in the second image set
      sumCorrs[1] += point[1] # sum of the y-coordinates of ...                   second ...

    averageOrgs = sumOrgs / len(self.orgs)
    averageCorrs = sumCorrs / len(self.Corrs)  
    
    return averageOrgs, averageCorrs
  
  # compute the scale s, which will be used in T_norm, T_norm s in two image are different
  def scale(self, points, average):
    sumOfDist = 0.0

    for point in points:
      sumOfDist += np.linalg.norm(point - average)
    
    meanOfDist = sumOfDist / len(points) # mean distance of all key points from centroid (in one image)
    s = math.sqrt(2) / meanOfDist # s is scale

    return s

  # compute T_norm
  def matrixT(self, s, average):
    Cx = average[0]
    Cy = average[1]
    T = np.float64([[s, 0, -s * Cx], 
                  [0, s, -s * Cy], 
                  [0, 0, 1]])
    return T
  
  # normalize keypoints
  def normalize(self, points, T):
    normalizedPoints = np.zeros((len(points), 3))
    for i, point in enumerate(points):
      normalizedPoints[i] = T @ point
    return normalizedPoints
  
  # Ah = 0 A: 2ix9, i is the number of point correspondences
  def matrixA(self, normalizedOrgs, normalizedCorrs):
    A = np.zeros((len(normalizedOrgs)*2, 9))
    for i in A.shape[0]:
      A[i]= np.float64([0, 0, 0, -normalizedOrgs[i][0], -normalizedOrgs[i][1], -1, 
                        normalizedCorrs[i][1] * normalizedOrgs[i][0], normalizedCorrs[i][1] * normalizedOrgs[i][1], normalizedCorrs[i][1]])
                      
      A[i+1] = np.float64([normalizedOrgs[i][0], normalizedOrgs[i][1], 1, 0, 0, 0, 
                           -normalizedCorrs[i][0] * normalizedOrgs[i][0], -normalizedCorrs[i][0]* normalizedOrgs[i][1], -normalizedCorrs[i][0]])
    return A
  
  def computeH(self):
    
    averageOrgs, averageCorrs = self.average()
    
    sOrgs = self.scale(self.orgs, averageOrgs)
    sCorrs = self.scale(self.corrs, averageCorrs)
    
    T_normOrgs = self.matrixT(sOrgs, averageOrgs)
    T_normCorrs = self.matrixT(sCorrs, averageCorrs)
    
    normalizedOrgs = self.normalize(self.orgs, T_normOrgs)
    normalizedCorrs = self.normalize(self.corrs, T_normCorrs)

    A = self.matrixA(normalizedOrgs, normalizedCorrs)

    U, S, VT = np.linalg.svd(A)
    V = VT.T
    h = V[:, -1]
    H_tilt = h.reshape(3, 3)
    H = np.linalg.inv(T_norm_corrs) @ H_tilt @ T_norm_orgs

    return H 
